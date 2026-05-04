# tests/test_model_config.py
import pytest
from transformers import AutoConfig

EXPECTED_CONFIGS = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 16,
        "hidden_size": 2048,
        "head_dim": 64,
        "rope_theta": 500000.0,
        "rope_type": "llama3",
        "rope_factor": 32.0,
        "rope_original_max_pos": 8192,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "head_dim": 128,
        "rope_theta": 500000.0,
        "rope_type": "llama3",
        "rope_factor": 8.0,
        "rope_original_max_pos": 8192,
    },
}

@pytest.mark.parametrize("model_id,expected", EXPECTED_CONFIGS.items())
def test_llama_config(model_id, expected):
    cfg = AutoConfig.from_pretrained(model_id)
    assert cfg.num_attention_heads == expected["num_attention_heads"]
    assert cfg.num_key_value_heads == expected["num_key_value_heads"]
    assert cfg.num_hidden_layers == expected["num_hidden_layers"]
    assert cfg.hidden_size == expected["hidden_size"]
    assert cfg.hidden_size // cfg.num_attention_heads == expected["head_dim"]
    assert cfg.rope_theta == expected["rope_theta"]
    assert cfg.rope_scaling["rope_type"] == expected["rope_type"]
    assert cfg.rope_scaling["factor"] == expected["rope_factor"]
    assert cfg.rope_scaling["original_max_position_embeddings"] == expected["rope_original_max_pos"]
