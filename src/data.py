"""Instance shape helpers + tokenizer glue used by both pipeline code and tests.

An instance is a dict with keys ``id`` / ``sys`` / ``query`` / ``answer`` /
``chunks``. ``chunks`` is a dict keyed by opaque chunk ID, each value
``{text, source, retrieval_rank}``. Chunk order inside the prompt is by
``retrieval_rank`` ascending — never rely on Python dict insertion order.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


VALID_SOURCES = {"contacts", "messages", "calendar", "email", "notes"}


def get_chunk_order(instance: dict) -> list[str]:
    """Chunk IDs sorted by retrieval_rank ascending (rank 1 = first in prompt)."""
    return sorted(
        instance["chunks"].keys(),
        key=lambda cid: instance["chunks"][cid]["retrieval_rank"],
    )


def get_chunk_text(instance: dict, chunk_id: str) -> str:
    return instance["chunks"][chunk_id]["text"]


def tokenize_instance(instance: dict, tokenizer, device) -> dict:
    """Tokenize an instance into tensors shaped to feed the run_* pipelines.

    Returns a dict whose keys match ``run_stale`` / ``run_gold`` / ``run_hybrid``
    parameter names, so callers do ``run_stale(model, **tokenize_instance(...))``.

    Each value is a ``[1, L]`` long tensor on ``device``. Empty strings tokenize to
    ``[1, 0]`` — downstream prefill helpers must handle that.

    ``add_special_tokens=False`` everywhere: we do not want a fresh BOS injected at
    each segment boundary. If a BOS is needed, callers prepend it into ``sys``.
    """
    def enc(text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=device)

    order = get_chunk_order(instance)
    return {
        "sys_ids":        enc(instance["sys"]),
        "chunk_ids_list": [enc(instance["chunks"][cid]["text"]) for cid in order],
        "query_ids":      enc(instance["query"]),
        "answer_ids":     enc(instance["answer"]),
    }


def validate_instance(inst: dict) -> None:
    """Raise ``ValueError`` if ``inst`` violates the dataset schema.

    Checks structural shape only — does not tokenize. Caller-side token-budget
    checks live in tests so they can use the loaded tokenizer fixture.

    Required keys: ``id`` / ``sys`` / ``query`` / ``answer`` / ``chunks``.
    ``id`` must be a non-empty str so sweep results can be traced back to the
    dataset row. Each chunk must have ``text`` (str), ``source``
    (one of ``VALID_SOURCES``), and ``retrieval_rank`` (int). Within a single
    instance, ranks must form a permutation of ``1..len(chunks)`` (no gaps, no
    duplicates) — otherwise prompt order is ill-defined.
    """
    iid = inst.get("id", "?")
    required = {"id", "sys", "query", "answer", "chunks"}
    missing = required - inst.keys()
    if missing:
        raise ValueError(f"instance {iid}: missing fields {sorted(missing)}")
    if not isinstance(inst["id"], str) or not inst["id"]:
        raise ValueError(f"instance {iid}: 'id' must be a non-empty str")
    for k in ("sys", "query", "answer"):
        if not isinstance(inst[k], str):
            raise ValueError(f"instance {iid}: {k!r} must be str, got {type(inst[k]).__name__}")
    chunks = inst["chunks"]
    if not isinstance(chunks, dict) or len(chunks) < 2:
        raise ValueError(
            f"instance {iid}: chunks must be a dict with ≥2 entries; got "
            f"{len(chunks) if isinstance(chunks, dict) else type(chunks).__name__}. "
            "Cross-chunk reasoning is the whole point — single-chunk instances "
            "don't stress-test stale cache."
        )
    ranks = []
    for cid, c in chunks.items():
        if not isinstance(c, dict):
            raise ValueError(f"instance {iid}: chunk {cid!r} not a dict")
        for k in ("text", "source", "retrieval_rank"):
            if k not in c:
                raise ValueError(f"instance {iid}: chunk {cid!r} missing {k!r}")
        if not isinstance(c["text"], str) or not c["text"]:
            raise ValueError(f"instance {iid}: chunk {cid!r} text must be a non-empty str")
        if c["source"] not in VALID_SOURCES:
            raise ValueError(
                f"instance {iid}: chunk {cid!r} source {c['source']!r} not in {sorted(VALID_SOURCES)}"
            )
        if not isinstance(c["retrieval_rank"], int):
            raise ValueError(f"instance {iid}: chunk {cid!r} retrieval_rank must be int")
        ranks.append(c["retrieval_rank"])
    if sorted(ranks) != list(range(1, len(chunks) + 1)):
        raise ValueError(
            f"instance {iid}: retrieval_rank values {sorted(ranks)} must be a permutation of "
            f"{list(range(1, len(chunks) + 1))} — gaps or duplicates break prompt ordering"
        )


def load_jsonl_instances(path: str | Path) -> list[dict]:
    """Read a JSONL dataset file. Each non-empty line is one instance dict.

    Validates every instance via ``validate_instance`` before returning, so a
    bad line surfaces at load time, not when the sweep is half-done.
    """
    path = Path(path)
    instances: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                inst = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            validate_instance(inst)
            instances.append(inst)
    return instances
