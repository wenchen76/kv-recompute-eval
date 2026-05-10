"""Phase 3 — HKVD @ r=15% monotonic on full DEV_INSTANCE.

With ``ppl_gold <= ppl_stale`` and r=0.15, hybrid should land inside the
interval (closed, with fp tolerance):

    ppl_gold  <=  ppl_hybrid  <=  ppl_stale

Above ``ppl_stale`` means we paid recompute cost but the hybrid hurt the model
— strong signal that recompute writeback or selection ranking is inverted.
Below ``ppl_gold`` is basically impossible unless one of the runs is wrong.

No strict recovery-ratio bound here: on the 1B dev model the gap between
gold and stale is small (see Phase 2 multi-chunk test), so the HKVD win is
close to noise. Recovery ratio is printed for observability; the real go/no-go
is the Phase 5 sweep on 8B. This test is correctness-only.

Parametrized over both HKVD variants — adding ``hkvd_gradual`` here keeps the
dynamic recompute path under the same correctness gate as the static ``hkvd``.
If gradual breaks ``r=0`` / ``r=1`` invariants or inverts ranking, this surfaces
it before the Phase 5 sweep.
"""
from __future__ import annotations

import math

import pytest

from src.config import DEVICE, PPL_TOL
from src.data import tokenize_instance
from src.experiment import run_gold, run_hybrid, run_stale
from tests.fixtures import DEV_INSTANCE


R = 0.15
HKVD_VARIANTS = ["hkvd", "hkvd_gradual"]


@pytest.mark.parametrize("strategy", HKVD_VARIANTS)
def test_hkvd_middle_r_monotonic(loaded_model, strategy):
    model, tokenizer = loaded_model

    tok = tokenize_instance(DEV_INSTANCE, tokenizer, device=DEVICE)

    nll_gold = run_gold(model, **tok)
    nll_stale = run_stale(model, **tok)
    nll_hybrid = run_hybrid(model, **tok, strategy=strategy, r=R)

    for name, v in (("gold", nll_gold), ("stale", nll_stale), ("hybrid", nll_hybrid)):
        assert math.isfinite(v), f"{name} NLL not finite: {v}"

    ppl_gold = math.exp(nll_gold)
    ppl_stale = math.exp(nll_stale)
    ppl_hybrid = math.exp(nll_hybrid)

    # Recovery ratio for observability. Only meaningful when stale > gold;
    # otherwise the denominator is near-zero noise and the ratio is junk.
    gap = ppl_stale - ppl_gold
    recovery = (ppl_stale - ppl_hybrid) / gap if gap > 1e-6 else float("nan")
    print(
        f"[{strategy}] ppl_gold={ppl_gold:.6f}  ppl_stale={ppl_stale:.6f}  "
        f"ppl_hybrid_r{R}={ppl_hybrid:.6f}  recovery={recovery:.3f}"
    )

    # Small absolute tolerance: the PPL_TOL widens with dtype, and a tiny
    # slack on ppl_stale catches fp noise when the gold/stale gap is small.
    tol = max(PPL_TOL, 0.01 * ppl_stale)

    assert ppl_hybrid <= ppl_stale + tol, (
        f"{strategy} @ r={R}: ppl_hybrid={ppl_hybrid:.4f} > ppl_stale={ppl_stale:.4f} "
        f"(tol {tol:.2e}). Hybrid worse than pure stale means the mix is "
        "hurting — check recompute writeback or the sign of the HKVD ranking."
    )
    assert ppl_hybrid >= ppl_gold - tol, (
        f"{strategy} @ r={R}: ppl_hybrid={ppl_hybrid:.4f} < ppl_gold={ppl_gold:.4f} "
        f"(tol {tol:.2e}). Hybrid beating gold is suspicious — likely a bug "
        "in one of run_gold / run_hybrid or in the cache-length bookkeeping."
    )
