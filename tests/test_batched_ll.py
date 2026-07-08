"""The batched LL must numerically match the per-name reference.

`compute_name_ll_batch` is a drop-in fast path for `compute_name_ll(...)[0]`; if
they ever diverge, the audit's verifier features (and every downstream metric)
silently change. This pins them together on a tiny CPU model.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.evaluation.pipeline.experimental.compute_ll_names import (
    compute_name_ll,
    compute_name_ll_batch,
)


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=128)
    model = LlamaForCausalLM(cfg).eval()
    return tok, model, "cpu"


NAMES = ["Donald Walker", "Margaret Johnson", "Amy Romero", "X", "Jean-Luc de la Croix", ""]
PROMPTS = ["Name: ", "Patient: "]


def test_batched_matches_per_name(tiny_model):
    tok, model, device = tiny_model
    for prompt in PROMPTS:
        ref = [compute_name_ll(prompt, str(n), tok, model, device)[0] for n in NAMES]
        got = compute_name_ll_batch(prompt, NAMES, tok, model, device, batch_size=4)
        assert len(got) == len(ref)
        for r, g, name in zip(ref, got, NAMES):
            assert abs(r - g) < 1e-3, f"mismatch for {name!r} under {prompt!r}: {r} vs {g}"


def test_batch_size_invariance(tiny_model):
    tok, model, device = tiny_model
    a = compute_name_ll_batch("Name: ", NAMES, tok, model, device, batch_size=1)
    b = compute_name_ll_batch("Name: ", NAMES, tok, model, device, batch_size=8)
    for x, y in zip(a, b):
        assert abs(x - y) < 1e-3


def test_padding_side_restored(tiny_model):
    tok, model, device = tiny_model
    tok.padding_side = "left"
    compute_name_ll_batch("Name: ", NAMES, tok, model, device, batch_size=2)
    assert tok.padding_side == "left"  # function restores caller's setting
