"""Centralize MODEL_ID / DEVICE / DTYPE selection so every script reads from env vars.

Laptop dev (default):   MODEL_ID=meta-llama/Llama-3.2-1B-Instruct DEVICE=cpu DTYPE=float32
GPU real sweep:         MODEL_ID=meta-llama/Llama-3.1-8B-Instruct DEVICE=cuda DTYPE=bfloat16
"""
from __future__ import annotations

import os
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
DEVICE = os.environ.get("DEVICE", "cpu")

_dtype_str = os.environ.get("DTYPE", "float32")
DTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[_dtype_str]

# PPL tolerance widens for bf16/fp16 (see Phase 1.3).
PPL_TOL = 1e-4 if DTYPE == torch.float32 else 1e-2


@lru_cache(maxsize=1)
def load_model_and_tokenizer():
    """Load model + tokenizer once per process; cached so tests share a single instance.

    Uses sdpa attention (see plan Phase 0.3) and the env-driven MODEL_ID/DEVICE/DTYPE.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
    )
    model.to(DEVICE)
    model.eval()
    return model, tokenizer
