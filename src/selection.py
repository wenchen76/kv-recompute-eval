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
    """Top-r% positions by layer-1 K divergence — the earliest layer that
    carries actual staleness signal.

    Divergence per position = L2 norm over batch × n_kv_heads × head_dim of
    ``(K_gold - K_stale)``. Same selection reused across all layers (paper
    insight: early-layer ranking correlates with later layers).

    **Why layer 1, not layer 0**: layer 0's K is purely a function of the
    token embedding and global position (no attention has happened yet), so
    K_gold ≡ K_stale at layer 0 up to fp noise (~1e-7). Ranking on noise
    selects garbage. Layer 1 is the first layer whose K depends on the
    layer-0 *attention* output, which is where stale (chunk-local attention)
    starts to diverge from gold (full-prefix attention).

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
    diff = (k_gold - k_stale).float()
    # L2 per position = sqrt(sum over B, H, D of squared diff). sqrt is
    # monotonic so topk on the squared sum would give the same ranking, but
    # the sqrt value is useful if we ever want to log / threshold on it.
    d = diff.pow(2).sum(dim=(0, 1, 3)).sqrt()  # [L_c]
    topk_idx = torch.topk(d, k).indices.tolist()
    return {start + int(i) for i in topk_idx}


def select_positions(
    strategy: str,
    r: float,
    *,
    chunk_range: tuple[int, int],
    chunk_offsets: Sequence[tuple[int, int]],
    kv_gold: DynamicCache | None = None,
    kv_stale: DynamicCache | None = None,
    seed: int = 0,
) -> set[int]:
    """Dispatch to one of ``random`` / ``first_r`` / ``hkvd``.

    ``kv_gold`` / ``kv_stale`` only required for ``hkvd``. ``seed`` only used
    by ``random``.
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
    raise ValueError(f"unknown strategy: {strategy!r}")
