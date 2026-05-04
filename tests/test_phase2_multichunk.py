"""Phase 2 (b)+(c) — Multi-chunk PPL + shape sanity on full DEV_INSTANCE.

Full 5-chunk DEV_INSTANCE (sys + chunk_0..chunk_4). This is the setting where
staleness actually bites: each chunk_i (i>0) in the stale path fails to attend
to sys and to chunks 0..i-1, so cross-chunk reasoning (query spans chunk_0's
date + chunk_2's content) degrades. Gates:

* (c) Shape consistency — per-layer stale prefix K/V shape == gold prefix K/V
  shape, and total seq length == len(sys) + sum(len(chunk_i)).
* (b) PPL sanity — 
    - both NLLs finite, both PPLs positive,
    - ``ppl_stale >= ppl_gold`` (staleness never helps),
    - ``ppl_stale / ppl_gold < 10x`` — a blow-up past this is almost always a
      position_ids or attention-mask bug, not "staleness being bad".

1B caveat (see plan Phase 2.4): on the dev model the ratio is small (often
1.1x–2x) because cross-chunk attention contributes less; this test is
correctness-only and is NOT a go/no-go signal for CacheBlend.
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


# 10x is generous enough that noise on the 1B model won't trip it, strict enough
# that a structural bug (wrong positions, wrong mask) will.
PPL_RATIO_BOUND = 10.0


def test_multichunk_shape_and_ppl_sanity(loaded_model):
    model, tokenizer = loaded_model

    tok = tokenize_instance(DEV_INSTANCE, tokenizer, device=DEVICE)

    assert tok["sys_ids"].shape[-1] > 0
    assert len(tok["chunk_ids_list"]) == 5, "DEV_INSTANCE should have 5 chunks"
    for i, c in enumerate(tok["chunk_ids_list"]):
        assert c.shape[-1] > 0, f"chunk {i} tokenized empty"
    assert tok["query_ids"].shape[-1] > 0
    assert tok["answer_ids"].shape[-1] > 0

    sys_len = tok["sys_ids"].shape[-1]
    chunk_lens = [c.shape[-1] for c in tok["chunk_ids_list"]]
    expected_prefix_len = sys_len + sum(chunk_lens)

    # Build both prefix caches at the sys+chunks scope (pre-query) so per-layer
    # shapes are directly comparable. Duplicates a few lines of run_stale but
    # it's the scope under test.
    prefix_ids = torch.cat([tok["sys_ids"], *tok["chunk_ids_list"]], dim=-1)
    kv_gold_prefix = prefill_gold(model, prefix_ids)

    kv_sys = prefill_chunk(model, tok["sys_ids"], global_start_pos=0)
    pos = sys_len
    kv_chunks = []
    for c_ids in tok["chunk_ids_list"]:
        kv_chunks.append(prefill_chunk(model, c_ids, global_start_pos=pos))
        pos += c_ids.shape[-1]
    kv_stale_prefix = concat_per_chunk_kvs([kv_sys, *kv_chunks], model.config)

    # Gate (c): per-layer shape match + total length.
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
        assert ks.shape[-2] == expected_prefix_len, (
            f"layer {li}: seq len {ks.shape[-2]} != sys+sum(chunks)={expected_prefix_len}"
        )
        for name, t in (("K_gold", kg), ("V_gold", vg), ("K_stale", ks), ("V_stale", vs)):
            assert torch.isfinite(t).all(), f"layer {li}: {name} has nan/inf"

    assert kv_stale_prefix.get_seq_length() == expected_prefix_len
    assert kv_gold_prefix.get_seq_length() == expected_prefix_len

    # Gate (b): PPL sanity. run_* return NLL; exponentiate for the ratio bound.
    nll_gold = run_gold(model, **tok)
    nll_stale = run_stale(model, **tok)

    assert math.isfinite(nll_gold), f"run_gold returned non-finite NLL: {nll_gold}"
    assert math.isfinite(nll_stale), f"run_stale returned non-finite NLL: {nll_stale}"

    ppl_gold = math.exp(nll_gold)
    ppl_stale = math.exp(nll_stale)
    assert ppl_gold > 0 and ppl_stale > 0
    print(f"ppl_gold={ppl_gold:.6f}  ppl_stale={ppl_stale:.6f}")

    # Staleness costs something — but tolerate equality on tiny 1B runs where the
    # signal is close to noise. A strict ``>`` would be flaky; strict ``<`` below
    # catches regressions in the other direction.
    assert ppl_stale >= ppl_gold - 1e-6, (
        f"ppl_stale ({ppl_stale:.3f}) unexpectedly below ppl_gold ({ppl_gold:.3f}); "
        "staleness should not help. Check that run_stale is actually using the "
        "stale prefix (not gold) before scoring."
    )

    ratio = ppl_stale / ppl_gold
    assert ratio < PPL_RATIO_BOUND, (
        f"Multi-chunk: ppl_stale/ppl_gold={ratio:.2f} exceeds {PPL_RATIO_BOUND}x "
        f"(ppl_gold={ppl_gold:.3f}, ppl_stale={ppl_stale:.3f}). "
        "Likely position_ids or attention-mask bug, not staleness scale."
    )
