"""Position-selection strategies for the hybrid cache (Phase 3).

Each strategy returns a set of **global** positions in the sys+chunks prefix
index space; those positions are sourced from gold by ``mix_kv``, the rest
from stale. Sys positions are never selected — sys KV is identical under both
paths, so it's a no-op either way and it keeps the selection scope honest.

* ``select_random`` — uniform-without-replacement baseline. Fixed seed. If the
  real strategies can't beat this, something is broken.
* ``select_first_r_per_chunk`` — first r-fraction of each chunk. No model
  forward needed; cheap.
* ``select_hkvd_first_layer`` — top-r% by layer-1 K-divergence. Simplified
  CacheBlend (paper observation: early-layer divergence ranking correlates
  highly with later layers, so one shared selection across layers works).
* ``select_hkvd_gradual`` — CacheBlend Fig. 9 multi-layer gradual filter.
  Iterates over layers 1..L-1, narrowing the candidate set with that layer's
  K-divergence; k decays from ``round(start_scale * r * N)`` at the first
  filter round to ``round(r * N)`` at the last layer. Picks positions that
  rank high-divergence across multiple layers, not just one.

``select_positions`` dispatches by name so ``run_hybrid`` can take a string.
"""
from __future__ import annotations

import random
from typing import Sequence

import torch
from transformers import DynamicCache


def _k_from_r(n_total: int, r: float) -> int:
    """Round ``r * n_total`` to an int, clamp to ``[0, n_total]``.

    Explicit so the three strategies share the same rounding rule — otherwise
    r=1.0 might not hit exactly n_total on some strategies due to accumulated
    rounding, and the endpoint invariants break.
    """
    assert 0.0 <= r <= 1.0, f"r must be in [0, 1], got {r}"
    k = int(round(r * n_total))
    return max(0, min(k, n_total))


def select_random(
    chunk_range: tuple[int, int],
    r: float,
    *,
    seed: int = 0,
) -> set[int]:
    """Uniformly sample ``k = round(r * chunks_len)`` positions without
    replacement from ``[chunk_range[0], chunk_range[1])``.
    """
    start, end = chunk_range
    n_total = end - start
    k = _k_from_r(n_total, r)
    if k == 0:
        return set()
    if k == n_total:
        return set(range(start, end))
    rng = random.Random(seed)
    return set(rng.sample(range(start, end), k))


def select_first_r_per_chunk(
    chunk_offsets: Sequence[tuple[int, int]],
    r: float,
) -> set[int]:
    """For each chunk ``(start, end)``, select the first ``round(r*chunk_len)``
    positions. Per-chunk rounding (not global), so r=1 → every chunk fully
    selected, r=0 → none.
    """
    selected: set[int] = set()
    for s, e in chunk_offsets:
        n = e - s
        k = _k_from_r(n, r)
        selected.update(range(s, s + k))
    return selected


@torch.no_grad()
def select_hkvd_first_layer(
    kv_gold: DynamicCache,
    kv_stale: DynamicCache,
    chunk_range: tuple[int, int],
    r: float,
) -> set[int]:
    """Top-r% positions by layer-1 KV divergence — the earliest layer that
    carries actual staleness signal.

    Divergence per position = ``‖(ΔK, ΔV)‖₂`` over batch × n_kv_heads ×
    head_dim — i.e. K and V are concatenated and treated as a single
    ``2 * head_dim`` feature vector per (batch, head, position). Equivalent
    to ``sqrt(‖ΔK‖² + ‖ΔV‖²)``; K usually dominates magnitude but V picks up
    "attention pattern correct, output value off" cases that K alone misses,
    which matter more in deeper layers (V's relative magnitude grows with
    depth). Same selection reused across all layers (paper insight: early-
    layer ranking correlates with later layers).

    **Why layer 1, not layer 0**: layer 0's K and V are pure functions of
    the token embedding and global position (no attention has happened yet),
    so they're bit-equal between gold and stale up to fp noise (~1e-7).
    Ranking on noise selects garbage. Layer 1 is the first layer whose KV
    depends on the layer-0 *attention* output, which is where stale (chunk-
    local attention) starts to diverge from gold (full-prefix attention).

    Cast to fp32 before the squared sum so the norm doesn't under/overflow on
    bf16 / fp16 caches — the ranking is what matters, not the absolute value.
    """
    start, end = chunk_range
    n_total = end - start
    k = _k_from_r(n_total, r)
    if k == 0:
        return set()
    if k == n_total:
        return set(range(start, end))

    k_gold = kv_gold.key_cache[1][:, :, start:end, :]   # [B, H, L_c, D]
    k_stale = kv_stale.key_cache[1][:, :, start:end, :]
    v_gold = kv_gold.value_cache[1][:, :, start:end, :]
    v_stale = kv_stale.value_cache[1][:, :, start:end, :]
    diff_k_sq = (k_gold - k_stale).float().pow(2).sum(dim=(0, 1, 3))  # [L_c]
    diff_v_sq = (v_gold - v_stale).float().pow(2).sum(dim=(0, 1, 3))
    # Combined L2 = ‖concat(ΔK, ΔV)‖₂. sqrt is monotonic so topk on the
    # squared sum would give the same ranking, but the sqrt value is useful
    # if we ever want to log / threshold on it.
    d = (diff_k_sq + diff_v_sq).sqrt()
    topk_idx = torch.topk(d, k).indices.tolist()
    return {start + int(i) for i in topk_idx}


@torch.no_grad()
def select_hkvd_gradual(
    kv_gold: DynamicCache,
    kv_stale: DynamicCache,
    chunk_range: tuple[int, int],
    r: float,
    *,
    start_scale: float = 1.2,
) -> list[set[int]]:
    """CacheBlend Fig. 9 gradual-filter HKVD selection — **per-layer**.

    Returns a list of length ``n_layers``; entry ``i`` is the set of positions
    that get gold KV at layer ``i``. The sets nest:
    ``layers[1] ⊇ layers[2] ⊇ ... ⊇ layers[L-1]``, with the per-layer fraction
    decaying linearly from ``start_scale * r`` at layer 1 to
    ``end_scale * r`` at the last layer, where ``end_scale = 2 - start_scale``
    so the per-layer mean is exactly ``r``.

    The matching mean is what makes the comparison against ``first_r`` and
    ``hkvd_first_layer`` honest: at the same ``r``, all three strategies
    recompute the same total number of (layer, position) entries. Without
    the symmetric schedule (e.g. flooring at ``k_target``), gradual would
    quietly spend a bigger compute budget than the others.

    This mirrors paper Fig. 9 directly: "use r₁% as the HKVD tokens on the
    second layer; pick r₂% from those for the third layer; and so forth";
    paper says "on average we want r%", which is the symmetric-schedule
    interpretation.

    Layer 0 always gets the empty set: layer-0 K and V are pure functions of
    token + position (no attention has happened yet), so they're bit-equal
    between gold and stale up to fp noise — mixing there is a no-op.

    Endpoints:
    * ``r=0`` → ``[∅] * n_layers`` → ``mix_kv`` returns stale.
    * ``r=1`` → ``[full] * n_layers`` → ``mix_kv`` returns gold.

    **Approximation vs. paper**: the paper recomputes KV at each layer using
    the partial hybrid cache built up from prior rounds, so layer-i
    "attention deviation" reflects the cache state mid-filter. We use the
    fully-prefilled ``kv_gold`` (full-attention) and ``kv_stale`` (chunk-local)
    throughout — i.e., we measure layer-i divergence assuming the IDEAL
    prior-layer cache. Doing the paper's exact recipe needs custom layer-by-
    layer forwards and falls outside this code's architecture.

    Args:
        start_scale: width of the round-1 candidate band relative to ``r``.
            Must be in ``[1.0, 2.0]`` so ``end_scale = 2 - start_scale`` is
            in ``[0.0, 1.0]``. Paper says "slightly higher than r" — ``1.2``
            is the default (band 1.2×r → 0.8×r, mean r); ``1.5`` was tried
            and is too aggressive ("slightly" reads more naturally as ~10–20%
            extra). ``1.0`` collapses to a flat ``r`` per layer at every
            round (no narrowing — same HKVD set picked independently each
            layer; useful as ablation).
    """
    assert 1.0 <= start_scale <= 2.0, (
        f"start_scale must be in [1.0, 2.0] so the symmetric end_scale stays "
        f"in [0.0, 1.0], got {start_scale}"
    )
    end_scale = 2.0 - start_scale
    start, end = chunk_range
    n_total = end - start
    n_layers = len(kv_gold.key_cache)
    k_target = _k_from_r(n_total, r)

    # Endpoint shortcuts — also keep the ``r=0`` / ``r=1`` invariants exactly.
    if k_target == 0:
        return [set() for _ in range(n_layers)]
    if k_target == n_total:
        full = set(range(start, end))
        return [full.copy() for _ in range(n_layers)]
    if n_layers <= 1:
        # Degenerate model with only layer 0 — no signal anywhere. No-op.
        return [set() for _ in range(n_layers)]

    device = kv_gold.key_cache[1].device
    candidates = torch.arange(start, end, device=device, dtype=torch.long)

    per_layer: list[set[int]] = [set() for _ in range(n_layers)]
    n_rounds = n_layers - 1  # one filter pass per layer 1..n_layers-1
    for round_idx in range(n_rounds):
        layer_idx = 1 + round_idx
        # Linear schedule: scale[0] = start_scale, scale[n_rounds-1] = end_scale.
        # Mean across rounds = (start + end)/2 = 1.0, so per-layer rate
        # averages to r exactly — apples-to-apples vs other strategies at
        # the same r.
        scale = (
            1.0
            if n_rounds == 1
            else start_scale - (start_scale - end_scale) * (round_idx / (n_rounds - 1))
        )
        k_i = _k_from_r(n_total, min(1.0, scale * r))
        # Cap by current candidate count — sets must nest, can't grow back.
        k_i = min(k_i, len(candidates))

        if k_i < len(candidates):
            kg = kv_gold.key_cache[layer_idx][:, :, candidates, :].float()
            ks = kv_stale.key_cache[layer_idx][:, :, candidates, :].float()
            vg = kv_gold.value_cache[layer_idx][:, :, candidates, :].float()
            vs = kv_stale.value_cache[layer_idx][:, :, candidates, :].float()
            # Combined K+V L2 — same metric as select_hkvd_first_layer.
            diff_k_sq = (kg - ks).pow(2).sum(dim=(0, 1, 3))  # [num_candidates]
            diff_v_sq = (vg - vs).pow(2).sum(dim=(0, 1, 3))
            d = (diff_k_sq + diff_v_sq).sqrt()
            topk_idx = torch.topk(d, k_i).indices
            candidates = candidates[topk_idx]
        # Snapshot whatever's surviving as this layer's HKVD set.
        per_layer[layer_idx] = {int(p) for p in candidates.tolist()}

    return per_layer


def select_positions(
    strategy: str,
    r: float,
    *,
    chunk_range: tuple[int, int],
    chunk_offsets: Sequence[tuple[int, int]],
    kv_gold: DynamicCache | None = None,
    kv_stale: DynamicCache | None = None,
    seed: int = 0,
) -> set[int] | list[set[int]]:
    """Dispatch to one of ``random`` / ``first_r`` / ``hkvd`` / ``hkvd_gradual``.

    Return type depends on the strategy:

    * ``random`` / ``first_r`` / ``hkvd`` → ``set[int]`` (one set, applied
      uniformly across all layers by ``mix_kv``).
    * ``hkvd_gradual`` → ``list[set[int]]`` of length ``n_layers`` (per-layer
      selection — the sets nest from widest at layer 1 to narrowest at the
      last layer).

    ``mix_kv`` accepts either shape, so callers don't need to special-case.

    ``kv_gold`` / ``kv_stale`` only required for the two ``hkvd*`` strategies.
    ``seed`` only used by ``random``.
    """
    if strategy == "random":
        return select_random(chunk_range, r, seed=seed)
    if strategy == "first_r":
        return select_first_r_per_chunk(chunk_offsets, r)
    if strategy == "hkvd":
        assert kv_gold is not None and kv_stale is not None, (
            "hkvd selection needs kv_gold and kv_stale"
        )
        return select_hkvd_first_layer(kv_gold, kv_stale, chunk_range, r)
    if strategy == "hkvd_gradual":
        assert kv_gold is not None and kv_stale is not None, (
            "hkvd_gradual selection needs kv_gold and kv_stale"
        )
        return select_hkvd_gradual(kv_gold, kv_stale, chunk_range, r)
    raise ValueError(f"unknown strategy: {strategy!r}")
