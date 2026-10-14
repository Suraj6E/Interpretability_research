import json
from pathlib import Path

import pytest
import torch

from interp import metrics
from interp.results import analyze_question

IOI = "When Mary and John went to the store, John gave a drink to"
QUESTIONS = Path(__file__).resolve().parent.parent / "questions" / "smoke_v0.json"


@pytest.mark.parametrize("layer", [0, 5, 11])
def test_head_outputs_add_up_to_attention_output(model, layer):
    """Guards against reading the wrong hook or weights: the per-head outputs we
    measure, plus the shared bias, must rebuild the layer's real attention output."""
    tokens = model.to_tokens(IOI)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
        rebuilt = metrics.head_outputs(model, cache, layer).sum(dim=0) + model.b_O[layer]
    actual = cache[f"blocks.{layer}.hook_attn_out"][0, -1]
    torch.testing.assert_close(rebuilt, actual, atol=1e-4, rtol=1e-4)


def test_activation_strength_shape_and_sign(model):
    _, _, cache = metrics.run_prompt(model, IOI, max_tokens=64)
    activation = metrics.activation_strength(model, cache)
    assert len(activation["heads"]) == 12 and all(len(row) == 12 for row in activation["heads"])
    assert len(activation["mlp"]) == 12
    assert all(v >= 0 for row in activation["heads"] for v in row)
    assert all(v >= 0 for v in activation["mlp"])


def test_single_token_answers(model):
    assert model.to_string(metrics.single_token_id(model, " Mary")) == " Mary"
    with pytest.raises(ValueError):
        metrics.single_token_id(model, " Eiffelization")


def test_prompt_length_limit(model):
    with pytest.raises(ValueError):
        metrics.run_prompt(model, "word " * 100, max_tokens=10)


def test_ioi_smoke_question_is_answered_correctly(model):
    """Pipeline sanity check, not a research result: GPT-2 small should prefer
    ' Mary' on the clean IOI prompt and ' John' on its twin (Wang et al. 2023)."""
    question = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"][0]
    result = analyze_question(model, question, "gpt2-small")
    assert result["logit_diff"]["clean"] > 0
    assert result["logit_diff"]["twin"] < 0
