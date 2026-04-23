"""Shared pytest fixtures.

The model is expensive to load (Llama-3.2-1B ~5GB in fp32), so we load it once per
session. Every test that needs (model, tokenizer) should depend on `loaded_model`.
"""
from __future__ import annotations

import pytest

from src.config import DEVICE, MODEL_ID, load_model_and_tokenizer


@pytest.fixture(scope="session")
def loaded_model():
    model, tokenizer = load_model_and_tokenizer()
    return model, tokenizer


@pytest.fixture(scope="session")
def prompt_answer_ids(loaded_model):
    """Fixed prompt + answer used across Phase 1 tests.

    Returns (prompt_ids, answer_ids) as [1, L] long tensors on the model's device.
    """
    _, tokenizer = loaded_model
    prompt = (
        "You are a helpful assistant. The user asks: "
        "What is the capital of France? Answer in one short sentence."
    )
    answer = " The capital of France is Paris."
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    # add_special_tokens=False so we don't re-inject BOS between prompt and answer.
    answer_ids = tokenizer(
        answer, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(DEVICE)
    return prompt_ids, answer_ids


__all__ = ["loaded_model", "prompt_answer_ids", "MODEL_ID"]
