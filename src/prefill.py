"""Prefill primitives.

Three flavors:

* ``prefill_gold`` — prefill a span with no prior cache, positions implicitly [0, L).
  Used on the full gold prompt (or any contiguous segment that starts at position 0).
* ``prefill_chunk`` — prefill a span with no prior cache but **explicit global
  position_ids**. Used in the stale path so that each chunk's RoPE rotation matches
  its position in the final prompt, even though the chunk doesn't attend to its
  predecessors. This is the source of staleness: wrong cross-chunk attention, not
  wrong positional encoding.
* ``online_prefill`` — prefill a span on top of an existing ``past_kv``. Used to roll
  the query forward on whatever prefix cache (stale / hybrid) was built.
"""
from __future__ import annotations

import torch
from transformers import DynamicCache


@torch.no_grad()
def prefill_gold(model, prompt_ids: torch.Tensor) -> DynamicCache:
    """Run ``prompt_ids`` in one forward pass and return the resulting KV cache.

    Convention used by ``compute_answer_ppl``: callers pass ``full_prompt[:, :-1]``
    (prompt minus its last token), so the decode pass can include that last token
    alongside the answer and produce a logit that scores answer[0].

    Args:
        prompt_ids: [B, L] long tensor on the model's device. L may be 0 → returns
            an empty ``DynamicCache``.
    """
    if prompt_ids.shape[-1] == 0:
        return DynamicCache()
    out = model(prompt_ids, use_cache=True)
    return out.past_key_values


@torch.no_grad()
def prefill_chunk(
    model,
    chunk_ids: torch.Tensor,
    global_start_pos: int,
) -> DynamicCache:
    """Prefill a chunk in isolation, but at its **global** position in the final prompt.

    Running the chunk with global ``position_ids`` means RoPE rotates K/V with the
    angles it will have in the concatenated cache. The chunk still doesn't attend
    to anything before it — that's the staleness this path is meant to measure.

    Args:
        chunk_ids: [B, L] long tensor on the model's device. L may be 0 → returns
            an empty ``DynamicCache`` (useful for the empty-sys case).
        global_start_pos: starting position of this chunk in the final prompt
            (sum of all preceding segment lengths).
    """
    L = chunk_ids.shape[-1]
    if L == 0:
        return DynamicCache()

    position_ids = torch.arange(
        global_start_pos,
        global_start_pos + L,
        device=chunk_ids.device,
    ).unsqueeze(0)

    out = model(chunk_ids, position_ids=position_ids, use_cache=True)
    return out.past_key_values


@torch.no_grad()
def online_prefill(
    model,
    input_ids: torch.Tensor,
    past_kv: DynamicCache,
    start_pos: int,
) -> DynamicCache:
    """Continue prefill on top of an existing cache.

    Used to roll the query forward on the stale/hybrid prefix — the query's KV
    reflects whatever context (correct or stale) the prefix carries.

    Args:
        input_ids: [B, L] long tensor. L may be 0 → returns ``past_kv`` unchanged.
        past_kv: cache produced by prior prefill. Mutated in place by the model's
            append, so the caller should not reuse it on a separate branch without
            copying first.
        start_pos: position of ``input_ids[0]`` in the final prompt. Must match
            ``past_kv.get_seq_length()`` — we accept it as a separate arg so callers
            document intent and we can assert they agree.
    """
    L = input_ids.shape[-1]
    if L == 0:
        return past_kv

    assert past_kv.get_seq_length() == start_pos, (
        f"online_prefill: start_pos={start_pos} but past_kv has "
        f"{past_kv.get_seq_length()} tokens already"
    )

    position_ids = torch.arange(
        start_pos, start_pos + L, device=input_ids.device
    ).unsqueeze(0)

    out = model(
        input_ids,
        past_key_values=past_kv,
        position_ids=position_ids,
        use_cache=True,
    )
    return out.past_key_values
