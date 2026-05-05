"""Run the Phase 5 selective-recompute sweep.

Default grid matches ``plan.md`` Phase 5:

    strategy in {random, first_r, hkvd}
    r in {0.0, 0.05, 0.10, 0.15, 0.20, 0.50, 1.0}

The runner writes an append-only JSONL file plus summary artifacts under
``results/``. It reuses per-instance gold/stale prefix caches so the sweep does
not rebuild the same prefix for every strategy/r cell.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch

from src.cache_ops import concat_per_chunk_kvs, mix_kv
from src.config import DEVICE, DTYPE, MODEL_ID, load_model_and_tokenizer
from src.data import load_jsonl_instances, tokenize_instance
from src.metrics import compute_answer_ppl
from src.prefill import online_prefill, prefill_chunk, prefill_gold
from src.selection import select_positions


DEFAULT_R_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.50, 1.0]
DEFAULT_STRATEGIES = ["random", "first_r", "hkvd"]


def _ppl(nll: float) -> float:
    return math.exp(nll)


def _recovery(ppl_gold: float, ppl_stale: float, ppl_hybrid: float) -> float | None:
    denom = ppl_stale - ppl_gold
    if abs(denom) < 1e-12:
        return None
    return (ppl_stale - ppl_hybrid) / denom


@torch.no_grad()
def _build_prefix_state(model, tok: dict) -> dict:
    """Build reusable per-instance caches and offsets."""
    sys_ids = tok["sys_ids"]
    chunk_ids_list = tok["chunk_ids_list"]

    kv_sys = prefill_chunk(model, sys_ids, global_start_pos=0)
    sys_len = sys_ids.shape[-1]
    pos = sys_len
    kv_chunks = []
    chunk_offsets: list[tuple[int, int]] = []
    for c_ids in chunk_ids_list:
        chunk_start = pos
        kv_chunks.append(prefill_chunk(model, c_ids, global_start_pos=pos))
        pos += c_ids.shape[-1]
        chunk_offsets.append((chunk_start, pos))

    full_prefix_ids = torch.cat([sys_ids, *chunk_ids_list], dim=-1)
    return {
        "prefix_end": pos,
        "chunk_range": (sys_len, pos),
        "chunk_offsets": chunk_offsets,
        "kv_prefix_stale": concat_per_chunk_kvs([kv_sys, *kv_chunks], model.config),
        "kv_prefix_gold": prefill_gold(model, full_prefix_ids),
    }


@torch.no_grad()
def _score_from_prefix(model, tok: dict, prefix_kv, prefix_end: int) -> float:
    """Score answer NLL after online-prefilling query[:-1] over a prefix cache.

    Mutates ``prefix_kv`` (online_prefill appends to its per-layer lists in
    place). Caller owns ownership: pass a freshly-allocated cache (e.g. one
    just returned by ``mix_kv``) or ``copy.deepcopy(...)`` of a shared cache.
    Doing the deepcopy at the call site lets us skip it when the cache is
    already exclusive to this scoring call (saves ~21 deepcopies per instance).
    """
    query_ids = tok["query_ids"]
    assert query_ids.shape[-1] >= 1, "query must have at least one token"
    kv = online_prefill(
        model,
        query_ids[:, :-1],
        past_kv=prefix_kv,
        start_pos=prefix_end,
    )
    return compute_answer_ppl(
        model,
        prompt_last_token=query_ids[:, -1:],
        answer_ids=tok["answer_ids"],
        past_kv=kv,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize(rows: list[dict]) -> dict:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["strategy"], row["r"])].append(row)

    cells = []
    for (strategy, r), group in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        recoveries = [g["recovery"] for g in group if g["recovery"] is not None]
        cells.append(
            {
                "strategy": strategy,
                "r": r,
                "n": len(group),
                "mean_ppl_gold": statistics.fmean(g["ppl_gold"] for g in group),
                "mean_ppl_stale": statistics.fmean(g["ppl_stale"] for g in group),
                "mean_ppl_hybrid": statistics.fmean(g["ppl_hybrid"] for g in group),
                "mean_recovery": statistics.fmean(recoveries) if recoveries else None,
            }
        )
    return {"cells": cells}


def _decision(summary: dict) -> str:
    """Three independent gates per plan.md Phase 5.4:

    * **GO** — hkvd @ r=15% mean recovery ≥ 0.90 (validates the strategy).
    * **NO-GO** — even the best strategy at r=50% fails to recover ≥ 0.50;
      means the task is too hard for any selection budget or the stale-path
      implementation is broken. Tested on best-of, not any-of: random often
      sits below 0.5 at r=50% by itself, that alone isn't a sweep problem.
    * **Weak signal** — random @ r=15% already ≥ 0.90; task too easy to
      stress cross-chunk attention.

    Missing cells (e.g. trimmed r grid) yield ``INCONCLUSIVE`` rather than
    silently being treated as NO-GO.
    """
    by_cell = {(c["strategy"], c["r"]): c for c in summary["cells"]}
    hkvd_015 = by_cell.get(("hkvd", 0.15), {}).get("mean_recovery")
    random_015 = by_cell.get(("random", 0.15), {}).get("mean_recovery")
    r50_by_strategy = {
        c["strategy"]: c["mean_recovery"]
        for c in summary["cells"]
        if c["r"] == 0.50 and c["mean_recovery"] is not None
    }

    flags: list[str] = []

    # Gate 1: GO — does HKVD reach the 0.90 bar at low budget.
    if hkvd_015 is None:
        flags.append("INCONCLUSIVE: no hkvd @ r=0.15 cell in sweep.")
    elif hkvd_015 >= 0.90:
        flags.append(f"GO: hkvd @ r=15% mean recovery {hkvd_015:.3f} ≥ 0.90.")
    else:
        flags.append(f"NOT-GO: hkvd @ r=15% mean recovery {hkvd_015:.3f} < 0.90.")

    # Gate 2: NO-GO — best-of-strategies at r=50% can't clear 0.50.
    if r50_by_strategy:
        best_s, best_v = max(r50_by_strategy.items(), key=lambda kv: kv[1])
        if best_v < 0.50:
            cells_str = ", ".join(f"{s}={v:.3f}" for s, v in sorted(r50_by_strategy.items()))
            flags.append(
                f"NO-GO: best r=50% recovery {best_v:.3f} (from {best_s}) < 0.50 "
                f"({cells_str}) — task too hard or stale-path implementation may be broken."
            )

    # Gate 3: weak-signal warning — task too easy.
    if random_015 is not None and random_015 >= 0.90:
        flags.append(
            f"WEAK SIGNAL: random @ r=15% already recovers {random_015:.3f} ≥ 0.90 "
            "— task may not stress cross-chunk attention."
        )

    return "\n".join(flags)


def _write_summary(path: Path, rows: list[dict], run_path: Path, elapsed_s: float) -> None:
    summary = _summarize(rows)
    lines = [
        "# Sweep Summary",
        "",
        f"- Run file: `{run_path}`",
        f"- Model: `{MODEL_ID}`",
        f"- Device: `{DEVICE}`",
        f"- DType: `{DTYPE}`",
        f"- Rows: {len(rows)}",
        f"- Wall time: {elapsed_s:.1f}s",
        "",
        "## Mean Metrics",
        "",
        "| strategy | r | n | ppl_gold | ppl_stale | ppl_hybrid | recovery |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        rec = cell["mean_recovery"]
        rec_s = "n/a" if rec is None else f"{rec:.4f}"
        lines.append(
            "| {strategy} | {r:.2f} | {n} | {mean_ppl_gold:.4f} | "
            "{mean_ppl_stale:.4f} | {mean_ppl_hybrid:.4f} | {rec} |".format(
                **cell, rec=rec_s
            )
        )
    lines.extend(["", "## Decision", "", _decision(summary), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_plot(path: Path, rows: list[dict]) -> None:
    summary = _summarize(rows)
    by_strategy: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for cell in summary["cells"]:
        if cell["mean_recovery"] is not None:
            by_strategy[cell["strategy"]].append((cell["r"], cell["mean_recovery"]))

    plt.figure(figsize=(7, 4.5))
    for strategy, points in sorted(by_strategy.items()):
        points = sorted(points)
        plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=strategy)
    plt.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="GO threshold")
    plt.xlabel("r")
    plt.ylabel("mean recovery")
    plt.title("Selective KV recompute recovery")
    plt.ylim(bottom=min(-0.1, plt.ylim()[0]), top=max(1.05, plt.ylim()[1]))
    plt.grid(True, alpha=0.25)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/synthetic_qa.jsonl")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--r-values", nargs="*", type=float, default=DEFAULT_R_VALUES)
    parser.add_argument("--strategies", nargs="*", default=DEFAULT_STRATEGIES)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    model, tokenizer = load_model_and_tokenizer()
    instances = load_jsonl_instances(args.dataset)
    if args.max_instances is not None:
        instances = instances[: args.max_instances]

    results_dir = Path(args.results_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = results_dir / f"run_{timestamp}.jsonl"

    rows: list[dict] = []
    for idx, inst in enumerate(instances, start=1):
        instance_started = time.perf_counter()
        tok = tokenize_instance(inst, tokenizer, device=DEVICE)
        prefix_state = _build_prefix_state(model, tok)

        # Deepcopy for gold/stale scoring: kv_prefix_gold and kv_prefix_stale
        # are reused across every (strategy, r) cell below (selection + mix_kv),
        # and _score_from_prefix mutates its argument by appending query KV.
        # kv_hybrid below is fresh-per-cell so it doesn't need deepcopy.
        nll_gold = _score_from_prefix(
            model,
            tok,
            copy.deepcopy(prefix_state["kv_prefix_gold"]),
            prefix_state["prefix_end"],
        )
        nll_stale = _score_from_prefix(
            model,
            tok,
            copy.deepcopy(prefix_state["kv_prefix_stale"]),
            prefix_state["prefix_end"],
        )
        ppl_gold = _ppl(nll_gold)
        ppl_stale = _ppl(nll_stale)

        for strategy in args.strategies:
            for r in args.r_values:
                cell_started = time.perf_counter()
                selected = select_positions(
                    strategy,
                    r,
                    chunk_range=prefix_state["chunk_range"],
                    chunk_offsets=prefix_state["chunk_offsets"],
                    kv_gold=prefix_state["kv_prefix_gold"],
                    kv_stale=prefix_state["kv_prefix_stale"],
                    seed=args.seed,
                )
                kv_hybrid = mix_kv(
                    prefix_state["kv_prefix_gold"],
                    prefix_state["kv_prefix_stale"],
                    selected,
                    model.config,
                )
                nll_hybrid = _score_from_prefix(
                    model, tok, kv_hybrid, prefix_state["prefix_end"]
                )
                ppl_hybrid = _ppl(nll_hybrid)
                rows.append(
                    {
                        "id": inst["id"],
                        "strategy": strategy,
                        "r": r,
                        "ppl_hybrid": ppl_hybrid,
                        "ppl_gold": ppl_gold,
                        "ppl_stale": ppl_stale,
                        "nll_hybrid": nll_hybrid,
                        "nll_gold": nll_gold,
                        "nll_stale": nll_stale,
                        "recovery": _recovery(ppl_gold, ppl_stale, ppl_hybrid),
                        "rouge_l": None,
                        "exact_match": None,
                        "wall_time_ms": round((time.perf_counter() - cell_started) * 1000, 3),
                    }
                )

        _write_jsonl(run_path, rows)
        rows.clear()
        print(
            f"[{idx}/{len(instances)}] {inst['id']} done in "
            f"{time.perf_counter() - instance_started:.1f}s",
            flush=True,
        )

    # Read back appended rows so summary exactly matches the run file.
    run_rows = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines()]
    elapsed_s = time.perf_counter() - started
    _write_summary(results_dir / "summary.md", run_rows, run_path, elapsed_s)
    _write_plot(results_dir / "recovery_curve.png", run_rows)
    print(f"wrote {run_path}")
    print(f"wrote {results_dir / 'summary.md'}")
    print(f"wrote {results_dir / 'recovery_curve.png'}")


if __name__ == "__main__":
    main()
