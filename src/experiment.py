"""End-to-end path runners: gold / stale / (later) hybrid.

All three consume the same tensor dict produced by ``tokenize_instance`` — call
sites look like ``run_stale(model, **tokenize_instance(inst, tok, device))``.

Each runner returns a single scalar (mean NLL over the answer tokens, equivalent
to log(PPL)).
"""
from __future__ import annotations

import torch

from src.cache_ops import concat_per_chunk_kvs
from src.metrics import compute_answer_ppl
from src.prefill import online_prefill, prefill_chunk, prefill_gold


@torch.no_grad()
def run_gold(
    model,
    sys_ids: torch.Tensor,
    chunk_ids_list: list[torch.Tensor],
    query_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> float:
    """Gold baseline: full prefill over [sys, chunks, query], then score answer.

    Keeps the same call signature as ``run_stale`` so the two are drop-in
    comparable. Internally it just concatenates everything into one prompt and
    defers to ``prefill_gold`` + ``compute_answer_ppl``.
    """
    prompt_ids = torch.cat([sys_ids, *chunk_ids_list, query_ids], dim=-1)
    assert prompt_ids.shape[-1] >= 1, "run_gold: empty prompt"

    past_kv = prefill_gold(model, prompt_ids[:, :-1])
    return compute_answer_ppl(
        model,
        prompt_last_token=prompt_ids[:, -1:],
        answer_ids=answer_ids,
        past_kv=past_kv,
    )


@torch.no_grad()
def run_stale(
    model,
    sys_ids: torch.Tensor,
    chunk_ids_list: list[torch.Tensor],
    query_ids: torch.Tensor,
    answer_ids: torch.Tensor,
) -> float:
    """Stale path (see plan Phase 2.0):

    1. Prefill ``sys`` at global positions [0, sys_len).
    2. Prefill each chunk_i independently at its global position — no cross-chunk
       attention, no attention to sys. This is the source of staleness.
    3. Concat sys + chunks along the sequence axis → stale prefix cache.
    4. Online-prefill ``query[:, :-1]`` on top of the stale prefix, so the query
       attends to the stale context (matches real RAG: query arrives online).
    5. Score ``answer`` with the last query token as the decode-starter.

    Step 4 deliberately holds back the last query token: ``compute_answer_ppl``
    takes it as ``prompt_last_token`` and concatenates it with the answer so that
    every answer token — including answer[0] — has a predicting logit.
    """
    assert query_ids.shape[-1] >= 1, "run_stale: need at least one query token"

    # Step 1: sys. prefill_chunk handles empty sys → empty cache, and uses
    # global_start_pos=0 which is trivially sys's own position anyway.
    kv_sys = prefill_chunk(model, sys_ids, global_start_pos=0)

    # Step 2: per-chunk independent prefill, accumulating global position.
    kv_chunks = []
    pos = sys_ids.shape[-1]
    for chunk_ids in chunk_ids_list:
        kv_chunks.append(prefill_chunk(model, chunk_ids, global_start_pos=pos))
        pos += chunk_ids.shape[-1]
    # pos now equals len(sys) + sum(len(chunk_i)) — the start position of the query.

    # Step 3: concat into prefix cache.
    kv_prefix = concat_per_chunk_kvs([kv_sys, *kv_chunks], model.config)

    # Step 4: online prefill query EXCEPT last token.
    query_prefill_ids = query_ids[:, :-1]
    kv_full = online_prefill(model, query_prefill_ids, past_kv=kv_prefix, start_pos=pos)

    # Step 5: score answer.
    return compute_answer_ppl(
        model,
        prompt_last_token=query_ids[:, -1:],
        answer_ids=answer_ids,
        past_kv=kv_full,
    )
