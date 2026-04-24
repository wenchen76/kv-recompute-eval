"""Gate A2 — Realistic sanity on sys + chunk_0 only.

Slices DEV_INSTANCE down to ``{sys, chunk_0}`` so the stale path has real
divergence (chunk_0 doesn't attend to sys) but the scale stays tractable. Unlike
Gate A1 this is NOT a numerical-equality test. It asserts:

* every K/V tensor in the stale and gold prefix caches is finite (no nan/inf),
* per-layer K/V shapes match between gold and stale,
* ``ppl_stale`` is positive and finite,
* ``ppl_stale / ppl_gold < 5x`` — loose upper bound to catch explosions; the
  real ratio on the 1B dev model is usually 1.1x–2x.

If this fails with Gate A1 green, the structural primitives are fine but
something numerical is wrong: dtype cast, attention mask on a specific layer,
mixed-precision overflow. Don't move to Phase 3 until this is green.
"""
from __future__ import annotations

import math

import torch

from src.cache_ops import concat_per_chunk_kvs
from src.config import DEVICE
from src.data import tokenize_instance
from src.experiment import run_gold, run_stale
from src.prefill import prefill_chunk, prefill_gold
from tests.fixtures import DEV_INSTANCE


# Wide enough to survive 1B noise, tight enough that a position_ids / mask bug
# blows past it. Real ratio on DEV_INSTANCE + 1B is ~1.1x–2x.
PPL_RATIO_BOUND = 5.0


def test_gate_a2_sys_plus_single_chunk(loaded_model):
    model, tokenizer = loaded_model

    sliced = {
        **DEV_INSTANCE,
        "chunks": {"chunk_0": DEV_INSTANCE["chunks"]["chunk_0"]},
    }
    tok = tokenize_instance(sliced, tokenizer, device=DEVICE)

    # Fixture sanity — catches tokenizer surprises (e.g. empty sys) before the
    # rest of the test starts measuring the wrong thing.
    assert tok["sys_ids"].shape[-1] > 0, "DEV_INSTANCE sys should tokenize to >0 tokens"
    assert len(tok["chunk_ids_list"]) == 1
    assert tok["chunk_ids_list"][0].shape[-1] > 0
    assert tok["query_ids"].shape[-1] > 0
    assert tok["answer_ids"].shape[-1] > 0

    # Build both prefix caches directly (scope: sys + chunk_0, pre-query) so we
    # can compare shape and finiteness layer-by-layer. run_gold / run_stale
    # bake in the ``prompt[:, :-1]`` slice, which would misalign the shapes.
    prefix_ids = torch.cat([tok["sys_ids"], *tok["chunk_ids_list"]], dim=-1)
    kv_gold_prefix = prefill_gold(model, prefix_ids)

    kv_sys = prefill_chunk(model, tok["sys_ids"], global_start_pos=0)
    kv_c0 = prefill_chunk(
        model,
        tok["chunk_ids_list"][0],
        global_start_pos=tok["sys_ids"].shape[-1],
    )
    kv_stale_prefix = concat_per_chunk_kvs([kv_sys, kv_c0], model.config)

    n_layers = model.config.num_hidden_layers
    assert len(kv_gold_prefix.key_cache) == n_layers
    assert len(kv_stale_prefix.key_cache) == n_layers

    for li in range(n_layers):
        kg, vg = kv_gold_prefix.key_cache[li], kv_gold_prefix.value_cache[li]
        ks, vs = kv_stale_prefix.key_cache[li], kv_stale_prefix.value_cache[li]

        assert ks.shape == kg.shape, (
            f"layer {li}: stale K shape {tuple(ks.shape)} != gold {tuple(kg.shape)}"
        )
        assert vs.shape == vg.shape, (
            f"layer {li}: stale V shape {tuple(vs.shape)} != gold {tuple(vg.shape)}"
        )
        for name, t in (("K_gold", kg), ("V_gold", vg), ("K_stale", ks), ("V_stale", vs)):
            assert torch.isfinite(t).all(), f"layer {li}: {name} has nan/inf"

    # Returned scalar is NLL (nats); PPL = exp(NLL). Compare in PPL space so the
    # 5x bound reads the same as the plan.
    nll_gold = run_gold(model, **tok)
    nll_stale = run_stale(model, **tok)

    assert math.isfinite(nll_gold), f"run_gold returned non-finite NLL: {nll_gold}"
    assert math.isfinite(nll_stale), f"run_stale returned non-finite NLL: {nll_stale}"

    ppl_gold = math.exp(nll_gold)
    ppl_stale = math.exp(nll_stale)
    assert ppl_gold > 0 and ppl_stale > 0
    print(f"ppl_gold={ppl_gold:.6f}  ppl_stale={ppl_stale:.6f}")

    ratio = ppl_stale / ppl_gold
    assert ratio < PPL_RATIO_BOUND, (
        f"Gate A2: ppl_stale/ppl_gold={ratio:.2f} exceeds {PPL_RATIO_BOUND}x "
        f"(ppl_gold={ppl_gold:.3f}, ppl_stale={ppl_stale:.3f}). "
        "Explosion points at numerical / mask / dtype bug, not staleness."
    )
