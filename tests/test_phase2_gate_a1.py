"""Gate A1 — degenerate-equality check for the stale path.

On ``SINGLE_CHUNK_INSTANCE`` (no sys, single chunk) chunk_0 has no prior context
to attend to, so stale-path KV and gold-path KV should match numerically. Any
discrepancy beyond PPL_TOL means a structural bug:

* concat direction wrong (e.g. cat-ing along head dim instead of seq dim),
* global position_ids wrong (chunk gets [0, L) instead of its global range — but
  those coincide here, so this specific test won't catch that; Gate A2 does),
* K/V tensor dim transposed,
* RoPE applied with flipped sign.

If this test fails, do not touch Gate A2 or Phase 3.
"""
from __future__ import annotations

import pytest

from src.config import DEVICE, PPL_TOL
from src.data import tokenize_instance
from src.experiment import run_gold, run_stale
from tests.fixtures import SINGLE_CHUNK_INSTANCE


def test_stale_equals_gold_on_single_chunk(loaded_model):
    model, tokenizer = loaded_model

    tok = tokenize_instance(SINGLE_CHUNK_INSTANCE, tokenizer, device=DEVICE)

    # Sanity on the fixture itself: empty sys tokenizes to [1, 0], single chunk
    # tokenizes to something non-empty. If either breaks, the rest of the test
    # is measuring the wrong thing.
    assert tok["sys_ids"].shape[-1] == 0, "SINGLE_CHUNK_INSTANCE.sys should tokenize to 0 tokens"
    assert len(tok["chunk_ids_list"]) == 1
    assert tok["chunk_ids_list"][0].shape[-1] > 0
    assert tok["query_ids"].shape[-1] > 0
    assert tok["answer_ids"].shape[-1] > 0

    ppl_gold = run_gold(model, **tok)
    ppl_stale = run_stale(model, **tok)
    print(f"ppl_gold={ppl_gold:.6f}  ppl_stale={ppl_stale:.6f}")

    assert ppl_stale == pytest.approx(ppl_gold, abs=PPL_TOL), (
        f"Gate A1 failed: ppl_stale={ppl_stale:.6f} vs ppl_gold={ppl_gold:.6f}; "
        f"diff {abs(ppl_stale - ppl_gold):.2e} exceeds tolerance {PPL_TOL:.1e}. "
        "Structural bug in stale path (concat / position_ids / shape / RoPE)."
    )
