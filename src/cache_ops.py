"""KV cache manipulation primitives.

Phase 2 only needs ``concat_per_chunk_kvs`` — assemble the stale prefix cache by
concatenating per-chunk DynamicCache objects along the sequence dimension. Phase 3
adds ``mix_kv`` (selective positions from gold vs stale).
"""
from __future__ import annotations

from typing import Sequence

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
