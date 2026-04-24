"""Instance shape helpers + tokenizer glue used by both pipeline code and tests.

An instance is a dict with keys ``sys`` / ``query`` / ``answer`` / ``chunks``.
``chunks`` is a dict keyed by opaque chunk ID, each value ``{text, source, retrieval_rank}``.
Chunk order inside the prompt is by ``retrieval_rank`` ascending — never rely on
Python dict insertion order.
"""
from __future__ import annotations

import torch


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
