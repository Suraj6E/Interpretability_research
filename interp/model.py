"""Load models. This is the only module that creates TransformerLens models, so a
different backend (e.g. nnsight for very large models) can be added here later.

TransformerLens 4 removed HookedTransformer; models are loaded as a TransformerBridge
with compatibility mode, which applies HookedTransformer's usual weight processing.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from transformer_lens.model_bridge import TransformerBridge
import transformer_lens.model_bridge.sources.transformers  # noqa: F401  (adds boot_transformers)

MODELS_FILE = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"

# Weight processing applied after loading. These change the coordinate system that
# component outputs are measured in (e.g. centering removes each output's mean, which
# the next LayerNorm would remove anyway), so they are recorded with every result.
PROCESSING = {
    "fold_ln": True,
    "center_writing_weights": True,
    "center_unembed": True,
    "fold_value_biases": True,
}


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short name used in configs and result paths, e.g. "gpt2-small"
    hf_repo: str  # Hugging Face repository
    revision: str  # exact Hugging Face commit hash
    dtype: str = "float32"


def model_spec(key: str, models_file: Path = MODELS_FILE) -> ModelSpec:
    """Look up a model's pinned repository and revision in configs/models.yaml."""
    models = yaml.safe_load(models_file.read_text(encoding="utf-8"))
    if key not in models:
        raise KeyError(f"{key!r} is not in {models_file.name}; known models: {', '.join(models)}")
    return ModelSpec(key=key, **models[key])


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(spec: ModelSpec, device: str | None = None) -> TransformerBridge:
    model = TransformerBridge.boot_transformers(
        spec.hf_repo,
        revision=spec.revision,
        device=device or default_device(),
        dtype=getattr(torch, spec.dtype),
    )
    model.enable_compatibility_mode(disable_warnings=True, **PROCESSING)
    model.eval()
    return model
