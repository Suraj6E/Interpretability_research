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


def test_dla_adds_up_to_logit_difference(model):
    """Every direct path into the logits (embeddings, each head, each MLP, the biases)
    must add up to the model's real logit difference, or DLA is computed wrongly."""
    correct = metrics.single_token_id(model, " Mary")
    wrong = metrics.single_token_id(model, " John")
    direction = metrics.logit_diff_direction(model, correct, wrong)
    with torch.no_grad():
        logits, cache = model.run_with_cache(model.to_tokens(IOI))
        dla = metrics.direct_logit_attribution(model, cache, direction)
        scale = cache["ln_final.hook_scale"][0, -1]

        def project(x):
            return ((x - x.mean(dim=-1, keepdim=True)) / scale) @ direction

        rest = (
            project(cache["hook_embed"][0, -1])
            + project(cache["hook_pos_embed"][0, -1])
            + project(model.b_O).sum()
            + model.b_U[correct] - model.b_U[wrong]
        )
    total = sum(map(sum, dla["heads"])) + sum(dla["mlp"]) + rest.item()
    assert total == pytest.approx(metrics.logit_diff(logits, correct, wrong), abs=1e-3)


def test_dla_signs_match_wang_et_al(model):
    """Known-result check: on IOI, name mover L9 H9 pushes toward the indirect object and
    negative name mover L10 H7 pushes away from it (Wang et al. 2023)."""
    question = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"][0]
    dla = analyze_question(model, question, "gpt2-small")["dla"]["heads"]
    assert dla[9][9] > 0
    assert dla[10][7] < 0


def test_attention_rows_add_up_to_one(model):
    _, _, cache = metrics.run_prompt(model, IOI, max_tokens=64)
    attention = metrics.attention_from_last(model, cache)
    assert len(attention) == 12 and len(attention[0]) == 12
    assert all(sum(row) == pytest.approx(1, abs=1e-3) for layer in attention for row in layer)


def test_greater_than_metric(model):
    question = {"id": "gt", "skill": "greater_than", "metric": "prob_diff", "year": 32,
                "clean": "The war lasted from the year 1732 to the year 17",
                "twin": "The war lasted from the year 1701 to the year 17"}
    result = analyze_question(model, question, "gpt2-small")
    assert -1 <= result["metric"]["twin"] < result["metric"]["clean"] <= 1


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
    assert result["metric"]["clean"] > 0
    assert result["metric"]["twin"] < 0
    assert result["top1"] == {"clean": " Mary", "twin": " John"}
