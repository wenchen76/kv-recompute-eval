"""End-to-end path runners: gold / stale / hybrid.

All three consume the same tensor dict produced by ``tokenize_instance`` — call
sites look like ``run_stale(model, **tokenize_instance(inst, tok, device))``.

Each runner returns a single scalar (mean NLL over the answer tokens, equivalent
to log(PPL)).
"""
from __future__ import annotations

import torch

from src.cache_ops import concat_per_chunk_kvs, mix_kv
from src.metrics import compute_answer_ppl
from src.prefill import online_prefill, prefill_chunk, prefill_gold
from src.selection import select_positions


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


@torch.no_grad()
def run_hybrid(
    model,
    sys_ids: torch.Tensor,
    chunk_ids_list: list[torch.Tensor],
    query_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    *,
    strategy: str,
    r: float,
    seed: int = 0,
) -> float:
    """Hybrid path (Phase 3):

    1. Build the stale prefix cache (sys + chunks) — same as ``run_stale`` steps 1–3.
    2. Build the gold prefix cache over the same (sys + chunks) span.
    3. Select positions within the chunk range via ``strategy`` / ``r``.
    4. Mix: selected positions come from gold, rest from stale → hybrid prefix.
    5. Online-prefill ``query[:, :-1]`` on top of the hybrid prefix.
    6. Score the answer with the held-back last query token.

    Endpoint invariants (asserted in Phase 3 tests):
    * ``r=0`` → empty selection → hybrid == stale → ppl_hybrid == ppl_stale.
    * ``r=1`` → full selection → hybrid == gold → ppl_hybrid == ppl_gold.

    Note: step 2 builds the full gold prefix so ``select_hkvd_layer0`` can read
    layer-0 K divergence. For strategies that don't need it (``random`` /
    ``first_r``) this is wasted work, but Phase 3 lives on a single dev instance
    and correctness beats micro-optimization here.
    """
    assert query_ids.shape[-1] >= 1, "run_hybrid: need at least one query token"

    # Step 1: stale prefix — sys + per-chunk, tracking each chunk's global span
    # for the first_r strategy.
    kv_sys = prefill_chunk(model, sys_ids, global_start_pos=0)
    sys_len = sys_ids.shape[-1]
    pos = sys_len
    kv_chunks: list = []
    chunk_offsets: list[tuple[int, int]] = []
    for c_ids in chunk_ids_list:
        chunk_start = pos
        kv_chunks.append(prefill_chunk(model, c_ids, global_start_pos=pos))
        pos += c_ids.shape[-1]
        chunk_offsets.append((chunk_start, pos))
    kv_prefix_stale = concat_per_chunk_kvs([kv_sys, *kv_chunks], model.config)
    prefix_end = pos
    chunk_range = (sys_len, prefix_end)

    # Step 2: gold prefix over the same sys+chunks span. Deliberately does not
    # include the query — the query is always online-prefilled on top.
    full_prefix_ids = torch.cat([sys_ids, *chunk_ids_list], dim=-1)
    kv_prefix_gold = prefill_gold(model, full_prefix_ids)

    # Step 3: selection — positions restricted to chunk range.
    selected = select_positions(
        strategy,
        r,
        chunk_range=chunk_range,
        chunk_offsets=chunk_offsets,
        kv_gold=kv_prefix_gold,
        kv_stale=kv_prefix_stale,
        seed=seed,
    )

    # Step 4: mix.
    kv_prefix_hybrid = mix_kv(kv_prefix_gold, kv_prefix_stale, selected, model.config)

    # Step 5: online prefill query except last token.
    query_prefill_ids = query_ids[:, :-1]
    kv_full = online_prefill(
        model, query_prefill_ids, past_kv=kv_prefix_hybrid, start_pos=prefix_end
    )

    # Step 6: score.
    return compute_answer_ppl(
        model,
        prompt_last_token=query_ids[:, -1:],
        answer_ids=answer_ids,
        past_kv=kv_full,
    )
