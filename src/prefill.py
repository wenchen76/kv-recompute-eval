"""Prefill helpers. Phase 1 has only gold prefill; per-chunk stale prefill lands in Phase 2."""
from __future__ import annotations

import torch
from transformers import DynamicCache


@torch.no_grad()
def prefill_gold(model, prompt_ids: torch.Tensor) -> DynamicCache:
    """Run the given tokens in one forward pass and return the resulting KV cache.

    Convention used by ``compute_answer_ppl``: callers pass ``full_prompt[:, :-1]``
    (i.e. prompt minus the last token), so that the decode pass can include that
    last token alongside the answer and produce a logit that scores answer[0].
    Feeding the full prompt here would make answer[0] unscorable in the prefill+decode
    path. See the call site in ``tests/test_gold_ppl.py`` for the detailed reason.

    Args:
        model: causal LM (returns .past_key_values when use_cache=True).
        prompt_ids: [B, L] long tensor on the model's device.

    Returns:
        DynamicCache with per-layer K,V of shape [B, n_kv_heads, L, head_dim].
    """
    out = model(prompt_ids, use_cache=True)
    return out.past_key_values
