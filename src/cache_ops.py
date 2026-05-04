"""KV cache manipulation primitives.

* ``concat_per_chunk_kvs`` — assemble the stale prefix cache by concatenating
  per-chunk DynamicCache objects along the sequence dimension (Phase 2).
* ``mix_kv`` — build a hybrid prefix: positions in ``selected_positions`` sourced
  from gold, everything else from stale (Phase 3).
"""
from __future__ import annotations

from typing import Sequence, Set

import torch
from transformers import DynamicCache


def _assert_layer_shape(t: torch.Tensor, expected: tuple[int, int, int, int], name: str) -> None:
    """Shape check that surfaces a useful error for GQA mistakes (n_kv_heads vs n_q_heads)
    and for head_dim model-family mismatches (1B=64, 8B=128). Hard-coding dims is the
    kind of bug this catches."""
    B, H, L, D = expected
    assert t.dim() == 4, f"{name}: expected 4-D [B, n_kv_heads, L, head_dim], got shape {tuple(t.shape)}"
    assert t.shape[0] == B, f"{name}: batch {t.shape[0]} != {B}"
    assert t.shape[1] == H, f"{name}: n_kv_heads {t.shape[1]} != {H} (GQA: did you pass n_q_heads by mistake?)"
    assert t.shape[3] == D, f"{name}: head_dim {t.shape[3]} != {D}"
    # seq len L is variable per-cache; caller checks totals separately.


def concat_per_chunk_kvs(
    caches: Sequence[DynamicCache],
    model_config,
) -> DynamicCache:
    """Concatenate a sequence of per-chunk DynamicCaches along the sequence axis.

    Each input cache was produced by ``prefill_chunk`` for one segment (sys or chunk).
    The returned cache is the stale **prefix** — ready to be extended online by the
    query via ``online_prefill``.

    Args:
        caches: list of DynamicCache in prompt order. Empty caches (e.g. empty sys)
            are skipped. At least one non-empty cache required.
        model_config: model.config. Used to derive expected ``n_kv_heads`` and
            ``head_dim`` for shape assertions — never hard-code these (1B has
            head_dim=64, 8B has 128).

    Returns:
        DynamicCache with per-layer K,V of shape [B, n_kv_heads, sum(L_i), head_dim]
        and ``_seen_tokens == sum(L_i)``.
    """
    non_empty = [c for c in caches if c.get_seq_length() > 0]
    assert non_empty, "concat_per_chunk_kvs: at least one non-empty cache required"

    n_layers = model_config.num_hidden_layers
    n_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads

    # Every non-empty cache must expose all layers. Use the first to read batch size.
    B = non_empty[0].key_cache[0].shape[0]

    merged = DynamicCache()
    total_len = 0
    for layer_idx in range(n_layers):
        per_layer_k = []
        per_layer_v = []
        for c in non_empty:
            k = c.key_cache[layer_idx]
            v = c.value_cache[layer_idx]
            L_i = k.shape[-2]
            _assert_layer_shape(k, (B, n_kv_heads, L_i, head_dim), f"K layer {layer_idx}")
            _assert_layer_shape(v, (B, n_kv_heads, L_i, head_dim), f"V layer {layer_idx}")
            per_layer_k.append(k)
            per_layer_v.append(v)
        merged.key_cache.append(torch.cat(per_layer_k, dim=-2))
        merged.value_cache.append(torch.cat(per_layer_v, dim=-2))
        if layer_idx == 0:
            total_len = merged.key_cache[0].shape[-2]

    merged._seen_tokens = total_len
    return merged


def mix_kv(
    kv_gold: DynamicCache,
    kv_stale: DynamicCache,
    selected_positions: Set[int],
    model_config,
) -> DynamicCache:
    """Build a hybrid prefix cache: positions in ``selected_positions`` sourced
    from ``kv_gold``, everything else from ``kv_stale``. Shared selection across
    all layers (per Phase 3 simplified HKVD — paper insight that early-layer
    divergence ranking correlates well with later layers).

    Sanity endpoints (Phase 3 tests assert these):
    * empty selection → output is bit-equal to ``kv_stale``,
    * full selection → output is bit-equal to ``kv_gold``.

    Args:
        kv_gold: full gold prefix cache (prefill on cat(sys, chunks)).
        kv_stale: stale prefix cache (concat of per-chunk prefills).
            Must cover the same sequence range as ``kv_gold``.
        selected_positions: global positions in ``[0, prefix_len)`` drawn from
            gold. Positions in the sys range are a no-op since sys KV is
            identical under both paths (sys has no prior context to attend to),
            but they're accepted — the caller doesn't have to special-case sys.
        model_config: model.config, for n_layers.

    Returns:
        DynamicCache with per-layer K,V of the shared prefix_len. Deep copy
        (via torch.where); mutating the result does not alias the inputs.
    """
    L_gold = kv_gold.get_seq_length()
    L_stale = kv_stale.get_seq_length()
    assert L_gold == L_stale, f"mix_kv: gold len {L_gold} != stale len {L_stale}"
    L = L_gold
    n_layers = model_config.num_hidden_layers

    device = kv_gold.key_cache[0].device
    mask = torch.zeros(L, dtype=torch.bool, device=device)
    if selected_positions:
        idx_list = sorted(selected_positions)
        assert idx_list[0] >= 0 and idx_list[-1] < L, (
            f"mix_kv: selected positions out of range [0, {L}): "
            f"min={idx_list[0]}, max={idx_list[-1]}"
        )
        idx = torch.tensor(idx_list, dtype=torch.long, device=device)
        mask[idx] = True
    # [1, 1, L, 1] broadcasts against K/V [B, n_kv_heads, L, head_dim].
    mask_b = mask.view(1, 1, L, 1)

    merged = DynamicCache()
    for li in range(n_layers):
        kg, vg = kv_gold.key_cache[li], kv_gold.value_cache[li]
        ks, vs = kv_stale.key_cache[li], kv_stale.value_cache[li]
        assert kg.shape == ks.shape, (
            f"layer {li}: gold K shape {tuple(kg.shape)} != stale {tuple(ks.shape)}"
        )
        assert vg.shape == vs.shape, (
            f"layer {li}: gold V shape {tuple(vg.shape)} != stale {tuple(vs.shape)}"
        )
        merged.key_cache.append(torch.where(mask_b, kg, ks))
        merged.value_cache.append(torch.where(mask_b, vg, vs))
    merged._seen_tokens = L
    return merged
