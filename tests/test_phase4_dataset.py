"""Phase 4 — synthetic QA dataset loader/validation.

Verifies ``data/synthetic_qa.jsonl`` is loadable, schema-clean, and tokenizes
within a sane budget for the dev-time 1B model. Heavy semantic checks
("does the answer actually require ≥2 chunks?") are out of scope — those are
caught indirectly by Phase 5: if a strategy gets full recovery at r=0 on an
instance, that instance is single-chunk and should be flagged.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import DEVICE
from src.data import VALID_SOURCES, load_jsonl_instances, tokenize_instance


DATASET_PATH = Path(__file__).parent.parent / "data" / "synthetic_qa.jsonl"

# Per plan.md Phase 4: chunks ~200 tokens, 5 per instance → prefix budget.
# 1B/CPU sub-second target sets the upper bound; the 8B sweep can absorb
# wider but no point asking for it. Keep slack for sys + query + answer.
MAX_PREFIX_TOKENS = 1500   # sys + 5 chunks + query
MAX_INSTANCE_TOKENS = 1800  # +answer


@pytest.fixture(scope="module")
def instances() -> list[dict]:
    if not DATASET_PATH.exists():
        pytest.skip(f"dataset not found at {DATASET_PATH}; run scripts/gen_dataset.py")
    return load_jsonl_instances(DATASET_PATH)


def test_dataset_nonempty(instances):
    assert len(instances) >= 10, (
        f"only {len(instances)} instances — Phase 5 sweep targets 50. "
        "Extend via scripts/gen_dataset.py."
    )


def test_unique_ids(instances):
    ids = [inst["id"] for inst in instances]
    assert len(ids) == len(set(ids)), "duplicate instance ids"


def test_chunk_count_in_realistic_range(instances):
    """Real retrievers return variable top-k; allow [2, 10]. Lower bound is
    cross-chunk reasoning. Upper bound is sweep cost — 10 chunks at ~150
    tokens already pushes the prefix to 1500."""
    for inst in instances:
        n = len(inst["chunks"])
        assert 2 <= n <= 10, f"{inst.get('id')}: {n} chunks outside [2, 10]"


def test_chunk_sources_all_valid(instances):
    """Sources may repeat within an instance (real RAG often returns 2+
    chunks from the same source — e.g. an email thread)."""
    for inst in instances:
        for cid, c in inst["chunks"].items():
            assert c["source"] in VALID_SOURCES, (
                f"{inst.get('id')}/{cid}: source {c['source']!r} not in {sorted(VALID_SOURCES)}"
            )


def test_dataset_source_coverage(instances):
    """Across the whole dataset, every source type should appear at least once
    — otherwise the sweep is biased toward whichever sources are present."""
    seen = {c["source"] for inst in instances for c in inst["chunks"].values()}
    missing = VALID_SOURCES - seen
    assert not missing, f"sources never used in dataset: {sorted(missing)}"


def test_sys_prompt_consistent(instances):
    """All instances share one sys prompt — sys KV gets prefilled once across the
    sweep and reused, so per-instance variation would just multiply prefill cost."""
    sys_strings = {inst["sys"] for inst in instances}
    assert len(sys_strings) == 1, (
        f"found {len(sys_strings)} distinct sys prompts; sweep assumes one"
    )


def test_token_budget(instances, loaded_model):
    """Each instance's prefix (sys + chunks + query) fits within budget so the
    dev-loop forward pass on 1B/CPU stays sub-second-ish."""
    _, tokenizer = loaded_model
    over_budget: list[tuple[str, int, int]] = []
    for inst in instances:
        tok = tokenize_instance(inst, tokenizer, device=DEVICE)
        prefix_len = (
            tok["sys_ids"].shape[-1]
            + sum(c.shape[-1] for c in tok["chunk_ids_list"])
            + tok["query_ids"].shape[-1]
        )
        full_len = prefix_len + tok["answer_ids"].shape[-1]
        if prefix_len > MAX_PREFIX_TOKENS or full_len > MAX_INSTANCE_TOKENS:
            over_budget.append((inst.get("id", "?"), prefix_len, full_len))
    assert not over_budget, (
        f"{len(over_budget)} instance(s) exceed budget "
        f"(prefix > {MAX_PREFIX_TOKENS} or full > {MAX_INSTANCE_TOKENS}): "
        f"{over_budget[:5]}"
    )


def test_chunks_render_in_retrieval_rank_order(instances, loaded_model):
    """``tokenize_instance`` must order chunks by retrieval_rank ascending — not
    by dict insertion. Cheap check: the first chunk_ids_list entry should
    tokenize to the rank-1 chunk's text."""
    _, tokenizer = loaded_model
    for inst in instances:
        tok = tokenize_instance(inst, tokenizer, device=DEVICE)
        rank1_cid = min(inst["chunks"], key=lambda cid: inst["chunks"][cid]["retrieval_rank"])
        rank1_text = inst["chunks"][rank1_cid]["text"]
        expected_ids = tokenizer.encode(rank1_text, add_special_tokens=False)
        actual_ids = tok["chunk_ids_list"][0].squeeze(0).tolist()
        assert actual_ids == expected_ids, (
            f"{inst.get('id')}: first chunk in tokenized output is not the "
            f"retrieval_rank=1 chunk ({rank1_cid})"
        )
