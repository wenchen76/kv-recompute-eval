"""KV cache manipulation primitives.

* ``concat_per_chunk_kvs`` — assemble the stale prefix cache by concatenating
  per-chunk DynamicCache objects along the sequence dimension (Phase 2).
* ``mix_kv`` — *oracle* hybrid: positions in ``selected_positions`` sourced
  directly from gold KV, everything else from stale. Used as an upper bound
  on selective-recompute quality (it pretends the recompute is lossless).
* ``cacheblend_recompute`` — *real* CacheBlend selective recompute with a
  static plan: layer-by-layer forward whose attention reads from the merged
  cache (fresh K/V at already-recomputed positions in this layer + stale K/V
  everywhere else). Predicts true CacheBlend quality at a fixed selection.
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


def mix_kv(
    kv_gold: DynamicCache,
    kv_stale: DynamicCache,
    selected_positions: Set[int] | Sequence[Set[int]],
    model_config,
) -> DynamicCache:
    """Build a hybrid prefix cache: per-position pick between ``kv_gold`` and
    ``kv_stale``, optionally with a different selection at every layer.

    Two call shapes:

    * ``selected_positions: Set[int]`` — uniform across layers (used by
      ``random`` / ``first_r`` / ``hkvd`` strategies). Same set applied to
      every layer of the cache.
    * ``selected_positions: Sequence[Set[int]]`` of length ``n_layers`` —
      per-layer (used by ``hkvd_gradual``: layer i takes its own selection
      that's a subset of layer i-1's). The sequence MUST be exactly
      ``model_config.num_hidden_layers`` long; index ``i`` is the selection
      applied at layer i.

    Sanity endpoints (Phase 3 tests assert these):
    * empty selection (every layer) → output bit-equal to ``kv_stale``,
    * full selection (every layer) → output bit-equal to ``kv_gold``.

    Args:
        kv_gold: full gold prefix cache (prefill on cat(sys, chunks)).
        kv_stale: stale prefix cache (concat of per-chunk prefills).
            Must cover the same sequence range as ``kv_gold``.
        selected_positions: see above. Positions in the sys range are no-ops
            since sys KV is identical under both paths.
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

    # Normalize to per-layer list. set/frozenset → replicate; otherwise it's
    # already a sequence and must match n_layers exactly.
    if isinstance(selected_positions, (set, frozenset)):
        per_layer: Sequence[Set[int]] = [selected_positions] * n_layers
    else:
        per_layer = list(selected_positions)
        assert len(per_layer) == n_layers, (
            f"mix_kv: per-layer selection length {len(per_layer)} != "
            f"n_layers {n_layers}"
        )

    device = kv_gold.key_cache[0].device
    merged = DynamicCache()
    for li in range(n_layers):
        sel = per_layer[li]
        # Build this layer's mask. Cheap (boolean of length L) and lets each
        # layer pick a different subset.
        mask = torch.zeros(L, dtype=torch.bool, device=device)
        if sel:
            idx_list = sorted(sel)
            assert idx_list[0] >= 0 and idx_list[-1] < L, (
                f"mix_kv layer {li}: selected positions out of range [0, {L}): "
                f"min={idx_list[0]}, max={idx_list[-1]}"
            )
            idx = torch.tensor(idx_list, dtype=torch.long, device=device)
            mask[idx] = True
        mask_b = mask.view(1, 1, L, 1)

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


@torch.no_grad()
def cacheblend_recompute(
    model,
    full_prefix_ids: torch.Tensor,
    kv_stale: DynamicCache,
    selected_positions: Set[int],
    model_config,
) -> DynamicCache:
    """Real CacheBlend selective recompute with a static plan.

    For each selected position, run a layer-by-layer forward where
    attention reads from the *merged* cache: fresh K/V at the selected
    positions (just-recomputed this layer) + stale K/V everywhere else.
    Non-selected positions keep their stale K/V untouched at every layer.
    The same selection set is applied to every layer.

    This reproduces CacheBlend's actual approximation error: at layer ℓ
    the recomputed h_i^(ℓ) is the output of attention over a stale-
    polluted cache, so the resulting k_i^(ℓ), v_i^(ℓ) are *not* equal to
    the full-prefill values. Errors compound across layers exactly as the
    paper acknowledges. Contrast with ``mix_kv``, which substitutes
    gold K/V directly and so upper-bounds the achievable quality.

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
            recomputed at every layer.
        model_config: ``model.config``.

    Returns:
        DynamicCache of length ``L``. Position ``i`` has fresh K/V at
        every layer iff ``i ∈ selected_positions``; otherwise stale K/V
        at every layer.
    """
    L = kv_stale.get_seq_length()
    assert full_prefix_ids.shape[-1] == L, (
        f"cacheblend_recompute: full_prefix_ids length {full_prefix_ids.shape[-1]} "
        f"!= kv_stale length {L}"
    )
    n_layers = model_config.num_hidden_layers

    # Always start from a deep copy of stale; at every layer we overwrite
    # the K/V for selected positions only.
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
    sel_idx = torch.tensor(sel_list, dtype=torch.long, device=device)
    S = sel_idx.numel()

    # Embed selected tokens and prep RoPE (cos/sin shared across layers
    # because the selected set is fixed).
    h = model.model.embed_tokens(full_prefix_ids.index_select(1, sel_idx))
    sel_pos_ids = sel_idx.unsqueeze(0)
    cos, sin = model.model.rotary_emb(h, sel_pos_ids)

    # Causal mask: token at S-index s (with global pos sel_idx[s]) attends
    # to cache positions [0, sel_idx[s]]. Built once, reused per layer.
    cache_pos = torch.arange(L, device=device)
    can_attend = cache_pos.unsqueeze(0) <= sel_idx.unsqueeze(1)  # [S, L]
    additive = torch.zeros((S, L), dtype=dtype, device=device)
    additive.masked_fill_(~can_attend, torch.finfo(dtype).min)
    additive = additive.view(1, 1, S, L)

    n_q_heads = model_config.num_attention_heads
    n_kv_heads = model_config.num_key_value_heads
    head_dim = getattr(
        model_config, "head_dim", model_config.hidden_size // n_q_heads
    )
    n_rep = n_q_heads // n_kv_heads
    scaling = head_dim ** -0.5

    for li, layer in enumerate(model.model.layers):
        residual = h
        h_ln = layer.input_layernorm(h)
        attn = layer.self_attn
        B = h_ln.shape[0]

        q = attn.q_proj(h_ln).view(B, S, n_q_heads, head_dim).transpose(1, 2)
        k = attn.k_proj(h_ln).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
        v = attn.v_proj(h_ln).view(B, S, n_kv_heads, head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        merged.key_cache[li].index_copy_(2, sel_idx, k)
        merged.value_cache[li].index_copy_(2, sel_idx, v)

        K_full = repeat_kv(merged.key_cache[li], n_rep)
        V_full = repeat_kv(merged.value_cache[li], n_rep)

        scores = torch.matmul(q, K_full.transpose(-1, -2)) * scaling
        scores = scores + additive
        attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_w, V_full)
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            B, S, n_q_heads * head_dim
        )
        attn_out = attn.o_proj(attn_out)

        h = residual + attn_out
        residual = h
        h = layer.post_attention_layernorm(h)
        h = layer.mlp(h)
        h = residual + h

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
    """Dynamic-deviation gradual selective recompute (paper Fig. 9 spirit),
    in a form that maps cleanly onto vLLM-style production deployment.

    **No precomputed full-prefill cache input.** The layer-1 ranking
    signal that earlier variants borrowed from a precomputed full-prefill
    layer-1 cache is recovered here by simply forwarding all chunk tokens
    through layers 0–1 — that forward already produces fresh layer-1 K/V
    at every chunk position, and ‖fresh − stale‖ at layer 1 is
    numerically equivalent to the layer-1 full-prefill-vs-stale signal
    (because at layer 1 the recompute output matches what full prefill
    would compute on chunk positions). This is the only ranking signal
    production CacheBlend can afford; obtaining a precomputed full-prefill
    cache at any layer would require having already done the very
    prefill we are trying to avoid.

    Algorithm:

    1. ``R = full chunk range`` (all ``n_total`` positions). Embed the
       chunk tokens.
    2. **Layer 0**: forward over all chunk tokens. No K/V writeback —
       layer-0 K/V is a pure function of token+position so stale already
       equals fresh up to fp noise.
    3. **Layer 1**: forward over all chunk tokens. Write fresh K/V at
       *every* chunk position into the merged cache (the layer is forced
       to do this work anyway to produce the deviation signal, so we
       keep the K/V — layer-1 cache then matches what full prefill would
       compute on chunks, at zero extra compute). Compute
       ``d_1 = ‖fresh − stale‖_layer1`` across all chunk positions, then
       narrow ``R`` to the top ``sched_k(0)`` by ``d_1`` for layer 2.
    4. **Layer ℓ ∈ [2, L-1]**:
       - Forward over ``R``. Write fresh K/V at ``R``.
       - Compute ``d_ℓ`` at ``R``.
       - If ℓ < L-1: narrow ``R`` to top ``sched_k(ℓ-1)`` by ``d_ℓ``.

    Schedule applies to layers 2..L-1 (``n_narrowed = n_layers - 2``
    entries): symmetric linear from ``start_scale * r * n_total`` at
    layer 2 to ``(2 - start_scale) * r * n_total`` at layer L-1, so the
    mean keep size over those narrowed layers is ``r * n_total``. Layers
    0 and 1 always process the full chunk range — that cost is the price
    of the layer-1 ranking signal in production.

    Endpoints:

    * ``r = 0`` → ``k_target = 0`` → returns clone of stale (skip
      recompute entirely; we do not even pay layer-1 forward cost).
    * ``r = 1`` → ``k_target = n_total`` → defer to ``cacheblend_recompute``
      with the full chunk set; matches a full prefill within numerical
      noise.

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
        ``DynamicCache`` of length ``L``. Layer 1 has fresh K/V at every
        chunk position; deeper layers have fresh K/V on the dynamically
        narrowed subset; sys positions stay stale at every layer.
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
        # Same rounding rule as src.selection._k_from_r — duplicated here
        # to keep cache_ops independent of selection's private API.
        return max(0, min(int(round(ratio * n_total)), n_total))

    # Initialize merged cache (clone of stale).
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
        # r → 1 fast path: full chunk set at every layer. Defer to the
        # static recompute so we don't hit the schedule's
        # ``min(1, scale * r)`` clamp asymmetry that would otherwise drop
        # ``k_ℓ`` below ``n_total`` at the tail of the ramp.
        full_set = set(range(start, end))
        return cacheblend_recompute(
            model, full_prefix_ids, kv_stale, full_set, model_config
        )

    end_scale = 2.0 - start_scale
    # Schedule covers layers 2..L-1; round 0 ↔ layer 2.
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

    # Start with the full chunk range. Layers 0 and 1 process every chunk
    # token; the layer-1 forward both produces the ranking signal and
    # populates layer-1 cache fresh-everywhere on chunks.
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

        # Rebuild RoPE + causal mask each layer because R can shrink.
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
            # Layer ≥ 1: write fresh K/V into cache at R. At layer 1, R is
            # still the full chunk range (no narrowing yet) — that is the
            # "renew layer-1 cache so it matches a full prefill on chunks"
            # At layer ≥ 2, R is the dynamically narrowed subset.
            merged.key_cache[li].index_copy_(2, R, k)
            merged.value_cache[li].index_copy_(2, R, v)

            stale_k = kv_stale.key_cache[li].index_select(2, R)
            stale_v = kv_stale.value_cache[li].index_select(2, R)
            dk = (k - stale_k).float().pow(2).sum(dim=(0, 1, 3))
            dv = (v - stale_v).float().pow(2).sum(dim=(0, 1, 3))
            d_layer = (dk + dv).sqrt()
        # Layer 0: skip writeback — layer-0 K/V is a pure function of
        # token + position, so stale already equals fresh up to fp noise.

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

        # Narrow R for the next layer (li + 1). Narrowing applies to
        # layers 2..L-1, so we narrow when li ∈ [1, L-2]. The schedule
        # round for the next layer is ``round_idx = (li + 1) - 2 = li - 1``.
        if li + 1 < n_layers and li >= 1 and d_layer is not None:
            round_idx = li - 1  # next layer = li+1; round 0 corresponds to layer 2
            k_next = min(sched_k(round_idx), S_cur)
            if k_next < S_cur:
                keep_local = torch.topk(d_layer, k_next).indices.sort().values
                R = R[keep_local]
                h = h[:, keep_local, :]

    return merged
