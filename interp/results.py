"""Build result dicts (format: planning.md §6) and save them.

The live server and batch runs both use these functions, so the viewer always
receives the same JSON shape whether it comes from a file or from the server.
"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch

from interp import __version__, metrics
from interp.model import PROCESSING, ModelSpec

SCHEMA_VERSION = 1
MAX_TOKENS = 256  # longer prompts are refused; component maps are measured at one position anyway


def analyze_prompt(model, prompt: str, model_key: str) -> dict:
    """Everything the viewer shows for one free-form prompt (live mode)."""
    tokens, logits, cache = metrics.run_prompt(model, prompt, max_tokens=MAX_TOKENS)
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model_key,
        "prompt": prompt,
        "tokens": metrics.str_tokens(model, tokens),
        "activation": metrics.activation_strength(model, cache),
        "top_predictions": metrics.top_predictions(model, logits),
    }


def analyze_question(model, question: dict, model_key: str) -> dict:
    """One question from a question file: the clean prompt's maps plus logit differences."""
    correct_id = metrics.single_token_id(model, question["correct"])
    wrong_id = metrics.single_token_id(model, question["wrong"])
    tokens, logits, cache = metrics.run_prompt(model, question["clean"], max_tokens=MAX_TOKENS)
    twin_tokens, twin_logits, _ = metrics.run_prompt(model, question["twin"], max_tokens=MAX_TOKENS)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": question["id"],
        "skill": question["skill"],
        "model": model_key,
        "clean": question["clean"],
        "twin": question["twin"],
        "correct": question["correct"],
        "wrong": question["wrong"],
        "tokens": metrics.str_tokens(model, tokens),
        "n_tokens": {"clean": tokens.shape[-1], "twin": twin_tokens.shape[-1]},
        "logit_diff": {
            "clean": metrics.logit_diff(logits, correct_id, wrong_id),
            "twin": metrics.logit_diff(twin_logits, correct_id, wrong_id),
        },
        "activation": metrics.activation_strength(model, cache),
        "top_predictions": metrics.top_predictions(model, logits),
    }


def run_metadata(spec: ModelSpec, config_path: Path, questions_path: Path, device: str) -> dict:
    """What is needed to trace a result back to its code, data and settings (REPRODUCIBILITY.md)."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": config_path.as_posix(),
        "questions": questions_path.as_posix(),
        "questions_sha256": hashlib.sha256(questions_path.read_bytes()).hexdigest(),
        "model": {"hf_repo": spec.hf_repo, "revision": spec.revision, "dtype": spec.dtype},
        "processing": PROCESSING,
        "device": device,
        "versions": {
            "interp": __version__,
            "transformer_lens": version("transformer-lens"),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
    }


def save_result(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['id']}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return path


def _git_commit() -> str:
    """Current commit hash, with '-dirty' if there are uncommitted changes (EXPERIMENT_LOG.md rule 2)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
