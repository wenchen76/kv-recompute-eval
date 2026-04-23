"""PPL measurement primitives.

Scoring convention: we score ALL L answer tokens. To make that work with the
prefill+decode path, the cache is prefilled on ``prompt_ids[:, :-1]`` and the decode
pass sees ``[prompt[-1], answer[0], ..., answer[L-1]]``. The decode output's
logits[t] predicts decode_input[t+1], so the first L logits predict answer[0..L-1]
one-to-one. See ``test_gold_ppl.py`` for the call-site explanation.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F


def _mean_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Cross-entropy averaged over all label positions. logits, labels aligned 1:1."""
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(),
        labels.reshape(-1),
        reduction="mean",
    )
    return loss.item()


@torch.no_grad()
def compute_answer_ppl(
    model,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    past_kv,
) -> float:
    """Mean NLL over all L answer tokens, given a cache prefilled on prompt[:, :-1].

    Args:
        prompt_ids: [B, P] full prompt (all P tokens; we slice [-1] internally).
        answer_ids: [B, L] answer.
        past_kv: DynamicCache of length ``P - 1`` — produced by
            ``prefill_gold(model, prompt_ids[:, :-1])``. Deep-copied before use so the
            caller's cache is not mutated by the in-place append inside the model.

    Returns:
        mean NLL (nats) over answer[0..L-1]. log(PPL) == mean NLL.
    """
    P = prompt_ids.shape[1]
    L = answer_ids.shape[1]
    assert P >= 1 and L >= 1

    cached_len = past_kv.key_cache[0].shape[-2]
    assert cached_len == P - 1, (
        f"past_kv has length {cached_len}, expected {P - 1}. "
        "Call prefill_gold(model, prompt_ids[:, :-1]) to produce it."
    )

    last_prefix = prompt_ids[:, -1:]                              # [B, 1]
    decode_input = torch.cat([last_prefix, answer_ids], dim=1)    # [B, L+1]

    # Global positions: last prefix token sits at P-1, then answer at P..P+L-1.
    position_ids = torch.arange(
        P - 1, P + L, device=prompt_ids.device
    ).unsqueeze(0)

    kv = copy.deepcopy(past_kv)
    out = model(
        decode_input,
        past_key_values=kv,
        position_ids=position_ids,
        use_cache=True,
    )
    # logits[:, t, :] predicts decode_input[:, t+1]. First L logits predict answer[0..L-1];
    # logits[:, L, :] predicts whatever would follow answer — discard.
    answer_logits = out.logits[:, :L, :]
    answer_labels = answer_ids
    return _mean_nll(answer_logits, answer_labels)


@torch.no_grad()
def compute_answer_ppl_full_forward(
    model,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> float:
    """Reference: run concat(prompt, answer) in a single forward and score all L answer tokens.

    Used by the Phase 1.3 sanity check to cross-verify the prefill+decode path.
    """
    P = prompt_ids.shape[1]
    L = answer_ids.shape[1]
    assert L >= 1

    full = torch.cat([prompt_ids, answer_ids], dim=1)
    out = model(full, use_cache=False)
    # logits at index (P-1) predicts answer[0]; (P+L-2) predicts answer[L-1].
    answer_logits = out.logits[:, P - 1 : P + L - 1, :]
    answer_labels = answer_ids
    return _mean_nll(answer_logits, answer_labels)
