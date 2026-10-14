"""Per-component measurements (definitions: docs/RESEARCH_PROTOCOL.md §2.3).

Component maps are returned as {"heads": [[...] per layer], "mlp": [...]}: a
layers x heads grid plus one value per layer, measured at one token position
(the last one by default).
"""

import torch

# Only the activations these measurements need are kept in the cache.
CACHED_HOOKS = ("attn.hook_z", "hook_mlp_out")


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


def logit_diff(logits, correct_id: int, wrong_id: int, pos: int = -1) -> float:
    """logit(correct) - logit(wrong): how strongly the model prefers the correct answer."""
    return (logits[0, pos, correct_id] - logits[0, pos, wrong_id]).item()


def single_token_id(model, text: str) -> int:
    """Token id of an answer, which must be exactly one token (RESEARCH_PROTOCOL.md §3 rule 1)."""
    ids = model.to_tokens(text, prepend_bos=False)[0]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens, not 1.")
    return ids[0].item()


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
