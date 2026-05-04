"""Phase 3 endpoint invariants — r=0 → stale, r=1 → gold.

For all three strategies:

* ``r=0`` → selection is empty → ``mix_kv`` returns a copy of the stale prefix
  → the whole hybrid path is numerically equivalent to ``run_stale``.
* ``r=1`` → selection covers every chunk position → ``mix_kv`` returns a copy
  of the gold prefix. The hybrid path is then equivalent to "full gold prefix +
  online-prefilled query", which is what ``run_gold`` does, just split into two
  forwards instead of one.

Both are free sanity gates for ``mix_kv`` (direction of torch.where), selection
rounding at the endpoints, and the run_hybrid wiring. Failure here points to
a swap of gold/stale args or a rounding bug (``r=1`` missing the last position).

Tolerance is ``PPL_TOL`` (1e-4 fp32 / 1e-2 bf16). r=0 should be bit-exact; r=1
can differ by a tiny amount because the gold path is split (prefix forward +
query forward) vs ``run_gold``'s single forward — same reason Gate A1 uses
PPL_TOL and not zero.
"""
from __future__ import annotations

import pytest

from src.config import DEVICE, PPL_TOL
from src.data import tokenize_instance
from src.experiment import run_gold, run_hybrid, run_stale
from tests.fixtures import DEV_INSTANCE


STRATEGIES = ["random", "first_r", "hkvd"]


@pytest.fixture(scope="module")
def dev_tok(loaded_model):
    _, tokenizer = loaded_model
    return tokenize_instance(DEV_INSTANCE, tokenizer, device=DEVICE)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_r0_equals_stale(loaded_model, dev_tok, strategy):
    model, _ = loaded_model

    nll_stale = run_stale(model, **dev_tok)
    nll_hybrid = run_hybrid(model, **dev_tok, strategy=strategy, r=0.0)

    assert nll_hybrid == pytest.approx(nll_stale, abs=PPL_TOL), (
        f"r=0 ({strategy}): hybrid NLL {nll_hybrid:.6f} != stale {nll_stale:.6f}, "
        f"diff {abs(nll_hybrid - nll_stale):.2e}. Empty selection should make "
        "mix_kv a no-op copy of the stale prefix."
    )


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_r1_equals_gold(loaded_model, dev_tok, strategy):
    model, _ = loaded_model

    nll_gold = run_gold(model, **dev_tok)
    nll_hybrid = run_hybrid(model, **dev_tok, strategy=strategy, r=1.0)

    assert nll_hybrid == pytest.approx(nll_gold, abs=PPL_TOL), (
        f"r=1 ({strategy}): hybrid NLL {nll_hybrid:.6f} != gold {nll_gold:.6f}, "
        f"diff {abs(nll_hybrid - nll_gold):.2e}. Full selection should make the "
        "hybrid prefix bit-equal to the gold prefix; any mismatch past fp noise "
        "means selection rounding dropped a position or mix_kv has gold/stale "
        "swapped."
    )
