"""Per-component measurements (definitions: docs/RESEARCH_PROTOCOL.md §2.3).

Component maps are returned as {"heads": [[...] per layer], "mlp": [...]}: a
layers x heads grid plus one value per layer, measured at one token position
(the last one by default).
"""

import torch

# Only the activations these measurements need are kept in the cache.
CACHED_HOOKS = ("attn.hook_z", "hook_mlp_out", "attn.hook_pattern", "ln_final.hook_scale")


def run_prompt(model, prompt: str, max_tokens: int):
    """Run the model on one prompt; return (tokens, logits, cache)."""
    tokens = model.to_tokens(prompt)
    if tokens.shape[-1] > max_tokens:
        raise ValueError(f"The prompt is {tokens.shape[-1]} tokens; the limit is {max_tokens}.")
    # no_grad rather than inference_mode: cached tensors are later combined with the
    # model's weights (e.g. W_O), which inference-mode tensors are not allowed to be.
    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name.endswith(CACHED_HOOKS)
        )
    return tokens, logits, cache


@torch.no_grad()
def head_outputs(model, cache, layer: int, pos: int = -1) -> torch.Tensor:
    """What each head adds to the residual stream at one position: hook_z x W_O.

    Shape [n_heads, d_model]. The attention output bias b_O is left out because no
    single head owns it; heads plus b_O add up to the layer's attention output.
    """
    z = cache[f"blocks.{layer}.attn.hook_z"][0, pos]  # [n_heads, d_head]
    return torch.einsum("hd,hdm->hm", z, model.W_O[layer])


def mlp_output(cache, layer: int, pos: int = -1) -> torch.Tensor:
    """What the layer's MLP adds to the residual stream at one position. Shape [d_model]."""
    return cache[f"blocks.{layer}.hook_mlp_out"][0, pos]


def activation_strength(model, cache, pos: int = -1) -> dict:
    """L2 norm of every head's and MLP's output. Correlational: big is not the same as important."""
    layers = range(model.cfg.n_layers)
    return {
        "heads": [head_outputs(model, cache, layer, pos).norm(dim=-1).tolist() for layer in layers],
        "mlp": [mlp_output(cache, layer, pos).norm().item() for layer in layers],
    }


@torch.no_grad()
def direct_logit_attribution(model, cache, direction: torch.Tensor, pos: int = -1) -> dict:
    """DLA: how much each head's and MLP's output pushes the logits along `direction`
    (e.g. W_U[correct] - W_U[wrong]), through the final LayerNorm of this run (protocol §2.3).

    Signed: positive pushes toward the correct answer, negative away from it.
    Correlational: it counts only direct effects, not effects through later layers.
    """
    scale = cache["ln_final.hook_scale"][0, pos]  # the final LayerNorm's divisor for this run

    def project(outputs):
        centered = outputs - outputs.mean(dim=-1, keepdim=True)  # what LayerNorm subtracts
        return (centered / scale) @ direction

    layers = range(model.cfg.n_layers)
    return {
        "heads": [project(head_outputs(model, cache, layer, pos)).tolist() for layer in layers],
        "mlp": [project(mlp_output(cache, layer, pos)).item() for layer in layers],
    }


def logit_diff_direction(model, correct_id: int, wrong_id: int) -> torch.Tensor:
    """The unembedding direction that logit(correct) - logit(wrong) measures."""
    return model.W_U[:, correct_id] - model.W_U[:, wrong_id]


def greater_than_direction(model, year_ids: list[int], year: int) -> torch.Tensor:
    """For greater-than: mean unembedding of years > YY minus mean of years <= YY.
    An approximation, because the task metric is a probability difference, not a logit
    difference (decision 2026-10-15, planning.md §12)."""
    years = model.W_U[:, year_ids]
    return years[:, year + 1 :].mean(dim=-1) - years[:, : year + 1].mean(dim=-1)


def attention_from_last(model, cache, pos: int = -1, decimals: int = 4) -> list:
    """For every head: how much the given position attends to each token, as
    [layer][head][token]; each row adds up to 1. Rounded to keep result files small."""
    # Rounded after conversion to Python floats: float32 values rounded in torch still
    # print with ~17 digits (0.1613 -> 0.16130000352859497).
    return [
        [[round(w, decimals) for w in row] for row in cache[f"blocks.{layer}.attn.hook_pattern"][0, :, pos, :].tolist()]
        for layer in range(model.cfg.n_layers)
    ]


def logit_diff(logits, correct_id: int, wrong_id: int, pos: int = -1) -> float:
    """logit(correct) - logit(wrong): how strongly the model prefers the correct answer."""
    return (logits[0, pos, correct_id] - logits[0, pos, wrong_id]).item()


def greater_than_prob_diff(logits, year_ids: list[int], year: int, pos: int = -1) -> float:
    """Hanna et al. 2023: P(next year > YY) - P(next year <= YY), over the tokens "00".."99"."""
    probs = logits[0, pos].float().softmax(dim=-1)[year_ids]
    return (probs[year + 1 :].sum() - probs[: year + 1].sum()).item()


class AnswerTokenError(ValueError):
    """An answer is not a single token for this model's tokenizer (protocol §3 rule 1)."""


def single_token_id(model, text: str) -> int:
    """Token id of an answer, which must be exactly one token (RESEARCH_PROTOCOL.md §3 rule 1)."""
    ids = model.to_tokens(text, prepend_bos=False)[0]
    if len(ids) != 1:
        raise AnswerTokenError(f"{text!r} is {len(ids)} tokens, not 1.")
    return ids[0].item()


def year_token_ids(model) -> list[int]:
    """Token ids of the two-digit years "00".."99" used by the greater-than task."""
    return [single_token_id(model, f"{year:02d}") for year in range(100)]


def top_token(model, logits, pos: int = -1) -> str:
    """The model's single most likely next token."""
    return model.to_string(logits[0, pos].argmax())


def str_tokens(model, tokens) -> list[str]:
    return model.to_str_tokens(tokens[0])


def top_predictions(model, logits, k: int = 5, pos: int = -1) -> list[dict]:
    """The model's k most likely next tokens with their probabilities."""
    probs = logits[0, pos].float().softmax(dim=-1)
    top = probs.topk(k)
    return [
        {"token": model.to_string(index), "prob": prob.item()}
        for prob, index in zip(top.values, top.indices)
    ]
