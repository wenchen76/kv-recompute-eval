"""KV cache manipulation primitives.

* ``concat_per_chunk_kvs`` — assemble the stale prefix cache by concatenating
  per-chunk DynamicCache objects along the sequence dimension (Phase 2).
* ``cacheblend_recompute`` — *real* CacheBlend selective recompute with a
  static plan: layer-by-layer forward whose attention reads from the merged
  cache (fresh K/V at already-recomputed positions in this layer + stale K/V
  everywhere else). When given a chunk range, layers 0 and 1 process the full
  chunk range before narrowing to the fixed selection, matching the gradual
  path's first-ranking-layer cost.
* ``cacheblend_recompute_gradual`` — dynamic-deviation gradual variant of
  the above (paper Fig. 9). Selection is fully interleaved with recompute:
  layers 0 and 1 process the full chunk range, the layer-1 forward
  produces the ranking signal as a side effect, and at each subsequent
  layer the next-layer keep set is chosen from that layer's just-computed
  fresh-vs-stale deviation. No ``kv_gold`` input — production CacheBlend
  cannot afford it.
"""
from __future__ import annotations

from typing import Sequence, Set

import torch
from transformers import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


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


@torch.no_grad()
def cacheblend_recompute(
    model,
    full_prefix_ids: torch.Tensor,
    kv_stale: DynamicCache,
    selected_positions: Set[int],
    model_config,
    *,
    chunk_range: tuple[int, int] | None = None,
) -> DynamicCache:
    """Real CacheBlend selective recompute with a static plan.

    For each selected position, run a layer-by-layer forward where attention
    reads from the *merged* cache: fresh K/V at the selected positions
    (just-recomputed this layer) + stale K/V everywhere else.

    If ``chunk_range`` is supplied, layers 0 and 1 forward the full chunk
    range, matching ``cacheblend_recompute_gradual``. Layer 1 writes fresh K/V
    for every chunk position, then later layers narrow back to
    ``selected_positions``. This keeps static strategies comparable with the
    gradual strategy when they are swept together.

    This reproduces CacheBlend's actual approximation error: at layer ℓ
    the recomputed h_i^(ℓ) is the output of attention over a stale-
    polluted cache, so the resulting k_i^(ℓ), v_i^(ℓ) are *not* equal to
    the full-prefill values. Errors compound across layers exactly as the
    paper acknowledges.

    Endpoints:

    * empty ``selected_positions`` → returns clone of stale.
    * every prefix position selected → matches a full prefill within
      attention-implementation numerical noise (eager math here vs the
      model's SDPA path).

    For the dynamic per-layer narrowing variant (paper Fig. 9), see
    ``cacheblend_recompute_gradual``.

    Args:
        model: ``LlamaForCausalLM``-compatible. Reads ``embed_tokens``,
            ``rotary_emb``, ``model.model.layers[*]``.
        full_prefix_ids: ``[1, L]`` long tensor of the full prefix
            (``cat(sys, *chunks)``). Length must match ``kv_stale``.
        kv_stale: stale prefix cache from ``concat_per_chunk_kvs``.
        selected_positions: global positions whose K/V should be
            recomputed after the layer-1 full-chunk warmup.
        model_config: ``model.config``.
        chunk_range: optional ``(start, end)`` global chunk-token span. When
            set, layer 1 is recomputed for this full range regardless of the
            static selection.

    Returns:
        DynamicCache of length ``L``. With ``chunk_range``, layer 1 has fresh
        K/V for all chunk positions, while layers 2+ have fresh K/V only at
        ``selected_positions``. Without ``chunk_range``, the historical static
        behavior is preserved.
    """
    L = kv_stale.get_seq_length()
    assert full_prefix_ids.shape[-1] == L, (
        f"cacheblend_recompute: full_prefix_ids length {full_prefix_ids.shape[-1]} "
        f"!= kv_stale length {L}"
    )
    n_layers = model_config.num_hidden_layers

    # Always start from a deep copy of stale; recompute overwrites selected
    # positions, plus the full chunk range at layer 1 when requested.
    merged = DynamicCache()
    for li in range(n_layers):
        merged.key_cache.append(kv_stale.key_cache[li].clone())
        merged.value_cache.append(kv_stale.value_cache[li].clone())
    merged._seen_tokens = L

    if not selected_positions:
        return merged

    sel_list = sorted(selected_positions)
    assert sel_list[0] >= 0 and sel_list[-1] < L, (
        f"cacheblend_recompute: selection out of range [0, {L}): "
        f"min={sel_list[0]}, max={sel_list[-1]}"
    )

    device = full_prefix_ids.device
    dtype = kv_stale.key_cache[0].dtype
    selected_set = set(sel_list)

    layer1_full_set: set[int] = set()
    if chunk_range is not None:
        chunk_start, chunk_end = chunk_range
        assert 0 <= chunk_start <= chunk_end <= L, (
            f"cacheblend_recompute: chunk_range {chunk_range} out of range [0, {L}]"
        )
        layer1_full_set = set(range(chunk_start, chunk_end))

    # To produce layer-1 K/V for the full chunk range, layer 0 must also
    # forward those positions so their layer-1 inputs exist. After layer 1,
    # the static path narrows back to the strategy-selected positions.
    active_list = sorted(selected_set | layer1_full_set)
    active_idx = torch.tensor(active_list, dtype=torch.long, device=device)
    sel_idx = torch.tensor(sel_list, dtype=torch.long, device=device)
    h = model.model.embed_tokens(full_prefix_ids.index_select(1, active_idx))
    cache_pos = torch.arange(L, device=device)

    n_q_heads = model_config.num_attention_heads
    n_kv_heads = model_config.num_key_value_heads
    head_dim = getattr(
        model_config, "head_dim", model_config.hidden_size // n_q_heads
    )
    n_rep = n_q_heads // n_kv_heads
    scaling = head_dim ** -0.5

    for li, layer in enumerate(model.model.layers):
        S_cur = active_idx.numel()
        sel_pos_ids = active_idx.unsqueeze(0)
        cos, sin = model.model.rotary_emb(h, sel_pos_ids)
        can_attend = cache_pos.unsqueeze(0) <= active_idx.unsqueeze(1)
        additive = torch.zeros((S_cur, L), dtype=dtype, device=device)
        additive.masked_fill_(~can_attend, torch.finfo(dtype).min)
        additive = additive.view(1, 1, S_cur, L)

        residual = h
        h_ln = layer.input_layernorm(h)
        attn = layer.self_attn
        B = h_ln.shape[0]

        q = attn.q_proj(h_ln).view(B, S_cur, n_q_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(h_ln).view(B, S_cur, n_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(h_ln).view(B, S_cur, n_kv_heads, head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        write_set = selected_set
        if li == 1 and layer1_full_set:
            write_set = selected_set | layer1_full_set
        write_idx = torch.tensor(sorted(write_set), dtype=torch.long, device=device)
        write_local = torch.searchsorted(active_idx, write_idx)
        merged.key_cache[li].index_copy_(
            2, write_idx, k.index_select(2, write_local)
        )
        merged.value_cache[li].index_copy_(
            2, write_idx, v.index_select(2, write_local)
        )

        K_full = repeat_kv(merged.key_cache[li], n_rep)
        V_full = repeat_kv(merged.value_cache[li], n_rep)

        scores = torch.matmul(q, K_full.transpose(-1, -2)) * scaling
        scores = scores + additive
        attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_w, V_full)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            B, S_cur, n_q_heads * head_dim
        )
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        h = residual + h

        if li == 1 and layer1_full_set and li + 1 < n_layers:
            keep_local = torch.searchsorted(active_idx, sel_idx)
            active_idx = sel_idx
            h = h[:, keep_local, :]

    return merged


@torch.no_grad()
def cacheblend_recompute_gradual(
    model,
    full_prefix_ids: torch.Tensor,
    kv_stale: DynamicCache,
    chunk_range: tuple[int, int],
    r: float,
    model_config,
    *,
    start_scale: float = 1.2,
) -> DynamicCache:
    """Dynamic CacheBlend recompute with gradual per-layer narrowing.

    This is the production-shaped ``hkvd_gradual`` path: it does not take a
    precomputed gold/full-prefill cache. Instead, it derives each ranking signal
    from the fresh K/V produced by the recompute itself:
    ``deviation = norm(fresh_kv - stale_kv)``.

    The current live position set is ``R``. It starts as the full chunk range.
    Layer 0 forwards all chunk tokens but does not write K/V back, because
    layer-0 K/V depends only on token and position. From layer 1 onward, each
    layer writes fresh K/V at ``R``, ranks those positions by fresh-vs-stale
    deviation, runs attention and MLP for all positions in ``R``, then narrows
    the layer output to form the next layer's input set.

    The keep schedule applies to layers 2..L-1. It linearly moves from
    ``start_scale * r * n_total`` to ``(2 - start_scale) * r * n_total``, so
    the average keep size across narrowed layers is ``r * n_total``. Layer 1
    always processes the full chunk range to produce the first deviation signal.

    Endpoints:
    * ``r = 0`` returns a clone of the stale cache without recompute.
    * ``r = 1`` defers to static full-chunk recompute for exact endpoint
      behavior.

    Args:
        model: ``LlamaForCausalLM``-compatible. Reads ``embed_tokens``,
            ``rotary_emb``, ``model.model.layers[*]``.
        full_prefix_ids: ``[1, L]`` long tensor of the full prefix
            (``cat(sys, *chunks)``). Length must match ``kv_stale``.
        kv_stale: stale prefix cache from ``concat_per_chunk_kvs``.
        chunk_range: ``(start, end)`` global positions where chunk tokens
            live (sys positions excluded — sys K/V is identical under
            stale and full-prefill paths and never selected).
        r: target keep ratio for the narrowed layers; mean over layers
            2..L-1 = ``r``.
        start_scale: schedule width; layer 2 keeps ``start_scale * r``,
            layer L-1 keeps ``(2 - start_scale) * r``. Must be in
            ``[1.0, 2.0]``. Default ``1.2``.

    Returns:
        ``DynamicCache`` of length ``L``. Layer 1 has fresh K/V for all chunk
        positions; deeper layers have fresh K/V for the dynamically narrowed
        subset. Sys positions remain stale because they are already identical
        under stale and full-prefill paths.
    """
    L = kv_stale.get_seq_length()
    assert full_prefix_ids.shape[-1] == L, (
        f"cacheblend_recompute_gradual: full_prefix_ids length "
        f"{full_prefix_ids.shape[-1]} != kv_stale length {L}"
    )
    assert 1.0 <= start_scale <= 2.0, (
        f"start_scale must be in [1.0, 2.0], got {start_scale}"
    )

    n_layers = model_config.num_hidden_layers
    start, end = chunk_range
    n_total = end - start

    def _k(ratio: float) -> int:
        # Match src.selection._k_from_r without depending on a private helper.
        return max(0, min(int(round(ratio * n_total)), n_total))

    # Start from stale and overwrite only positions that are recomputed.
    merged = DynamicCache()
    for li in range(n_layers):
        merged.key_cache.append(kv_stale.key_cache[li].clone())
        merged.value_cache.append(kv_stale.value_cache[li].clone())
    merged._seen_tokens = L

    if n_total == 0 or n_layers <= 1:
        return merged

    k_target = _k(r)
    if k_target == 0:
        return merged
    if k_target == n_total:
        # Preserve the r=1 endpoint exactly; the gradual schedule would otherwise
        # shrink at the low end of the symmetric ramp.
        full_set = set(range(start, end))
        return cacheblend_recompute(
            model,
            full_prefix_ids,
            kv_stale,
            full_set,
            model_config,
            chunk_range=chunk_range,
        )

    end_scale = 2.0 - start_scale
    # Round 0 chooses the keep set for layer 2; the final round targets L-1.
    n_narrowed = max(1, n_layers - 2)

    def sched_k(round_idx: int) -> int:
        scale = (
            1.0
            if n_narrowed == 1
            else start_scale - (start_scale - end_scale) * (round_idx / (n_narrowed - 1))
        )
        return _k(min(1.0, scale * r))

    device = full_prefix_ids.device
    dtype = kv_stale.key_cache[0].dtype

    # R is the current live chunk-position set. It shrinks after layer 1 and
    # may shrink again after each later non-final layer.
    R = torch.arange(start, end, dtype=torch.long, device=device).contiguous()
    h = model.model.embed_tokens(full_prefix_ids.index_select(1, R))

    n_q_heads = model_config.num_attention_heads
    n_kv_heads = model_config.num_key_value_heads
    head_dim = getattr(
        model_config, "head_dim", model_config.hidden_size // n_q_heads
    )
    n_rep = n_q_heads // n_kv_heads
    scaling = head_dim ** -0.5
    cache_pos = torch.arange(L, device=device)

    for li, layer in enumerate(model.model.layers):
        S_cur = R.numel()
        if S_cur == 0:
            break

        # R can shrink between layers, so RoPE positions and the causal mask are
        # rebuilt for the current live set.
        sel_pos_ids = R.unsqueeze(0)
        cos, sin = model.model.rotary_emb(h, sel_pos_ids)
        can_attend = cache_pos.unsqueeze(0) <= R.unsqueeze(1)  # [S_cur, L]
        additive = torch.zeros((S_cur, L), dtype=dtype, device=device)
        additive.masked_fill_(~can_attend, torch.finfo(dtype).min)
        additive = additive.view(1, 1, S_cur, L)

        residual = h
        h_ln = layer.input_layernorm(h)
        attn = layer.self_attn
        B = h_ln.shape[0]
        q = attn.q_proj(h_ln).view(B, S_cur, n_q_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(h_ln).view(B, S_cur, n_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(h_ln).view(B, S_cur, n_kv_heads, head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        d_layer: torch.Tensor | None = None
        if li > 0:
            # Commit fresh K/V for this layer. At layer 1, R is
            # still the full chunk range; at deeper layers, it is the previous
            # layer's survivors.
            merged.key_cache[li].index_copy_(2, R, k)
            merged.value_cache[li].index_copy_(2, R, v)

            # Rank the current live positions by the deviation just produced by
            # this layer's recompute.
            stale_k = kv_stale.key_cache[li].index_select(2, R)
            stale_v = kv_stale.value_cache[li].index_select(2, R)
            dk = (k - stale_k).float().pow(2).sum(dim=(0, 1, 3))
            dv = (v - stale_v).float().pow(2).sum(dim=(0, 1, 3))
            d_layer = (dk + dv).sqrt()
        # Layer 0 has no useful deviation signal and no K/V writeback.

        # Attention reads from the merged full-prefix cache for every position
        # currently in R.
        K_full = repeat_kv(merged.key_cache[li], n_rep)
        V_full = repeat_kv(merged.value_cache[li], n_rep)
        scores = torch.matmul(q, K_full.transpose(-1, -2)) * scaling
        scores = scores + additive
        attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_w, V_full)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            B, S_cur, n_q_heads * head_dim
        )
        attn_out = attn.o_proj(attn_out)

        # Finish this layer for every position in R before choosing which layer
        # outputs continue to the next layer.
        h = residual + attn_out
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        h = residual + h

        # Narrow after the layer output. The selected h rows are exactly the
        # input states for the next layer; K/V already written at the wider R
        # remains in this layer's cache.
        if li >= 1 and li + 1 < n_layers and d_layer is not None:
            round_idx = li - 1
            k_next = min(sched_k(round_idx), S_cur)
            if k_next < S_cur:
                keep_local = torch.topk(d_layer, k_next).indices.sort().values
                R = R[keep_local]
                h = h[:, keep_local, :]

    return merged
