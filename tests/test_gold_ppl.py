"""Phase 1.3 sanity check: full-forward PPL == prefill_gold + compute_answer_ppl.

If this test doesn't pass, the position_ids or logits offset is wrong — don't go to Phase 2.
"""
from __future__ import annotations

import pytest

from src.config import PPL_TOL
from src.metrics import compute_answer_ppl, compute_answer_ppl_full_forward
from src.prefill import prefill_gold


def test_full_forward_matches_prefill_decode(loaded_model, prompt_answer_ids):
    model, _ = loaded_model
    prompt_ids, answer_ids = prompt_answer_ids

    nll_full = compute_answer_ppl_full_forward(model, prompt_ids, answer_ids)

    # Why prompt_ids[:, :-1] (not the full prompt):
    # In the decode pass that scores the answer, logits[t] predicts decode_input[t+1].
    # If the cache covered the full prompt, the decode input would start with answer[0]
    # and no logit would predict it (we'd lose one token of the NLL average). Instead we
    # prefill one token short, then compute_answer_ppl prepends prompt[-1] to the decode
    # input so every answer token — including answer[0] — has a predicting logit.
    past_kv = prefill_gold(model, prompt_ids[:, :-1])
    nll_prefill = compute_answer_ppl(model, prompt_ids, answer_ids, past_kv)

    assert nll_full == pytest.approx(nll_prefill, abs=PPL_TOL), (
        f"full-forward NLL {nll_full:.6f} vs prefill+decode NLL {nll_prefill:.6f}; "
        f"diff {abs(nll_full - nll_prefill):.2e} exceeds tolerance {PPL_TOL:.1e}. "
        "Likely cause: position_ids offset or logits shift is wrong."
    )
