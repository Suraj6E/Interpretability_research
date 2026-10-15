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

SCHEMA_VERSION = 2
MAX_TOKENS = 256  # longer prompts are refused; component maps are measured at one position anyway


def analyze_prompt(
    model, prompt: str, model_key: str, correct: str | None = None, wrong: str | None = None
) -> dict:
    """Everything the viewer shows for one free-form prompt (live mode).

    DLA compares two answer tokens. If they are not given, the model's top two guesses are
    used. Raises metrics.AnswerTokenError if a given answer is not a single token.
    """
    tokens, logits, cache = metrics.run_prompt(model, prompt, max_tokens=MAX_TOKENS)
    top = metrics.top_predictions(model, logits)
    correct = correct or top[0]["token"]
    wrong = wrong or top[1]["token"]
    direction = metrics.logit_diff_direction(
        model, metrics.single_token_id(model, correct), metrics.single_token_id(model, wrong)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model": model_key,
        "prompt": prompt,
        "correct": correct,
        "wrong": wrong,
        "tokens": metrics.str_tokens(model, tokens),
        "activation": metrics.activation_strength(model, cache),
        "dla": metrics.direct_logit_attribution(model, cache, direction),
        "attention": metrics.attention_from_last(model, cache),
        "top_predictions": top,
    }


def analyze_question(model, question: dict, model_key: str) -> dict:
    """One question from a question file: the clean prompt's maps, the task metric on the
    clean and twin prompts, and the model's top guess on each.

    Raises metrics.AnswerTokenError if an answer is not a single token (protocol §3 rule 1).
    """
    measure, direction = _task(model, question)
    tokens, logits, cache = metrics.run_prompt(model, question["clean"], max_tokens=MAX_TOKENS)
    twin_tokens, twin_logits, _ = metrics.run_prompt(model, question["twin"], max_tokens=MAX_TOKENS)
    return {
        "schema_version": SCHEMA_VERSION,
        "id": question["id"],
        "skill": question["skill"],
        "model": model_key,
        "clean": question["clean"],
        "twin": question["twin"],
        **{key: question[key] for key in ("correct", "wrong", "year") if key in question},
        "tokens": metrics.str_tokens(model, tokens),
        "n_tokens": {"clean": tokens.shape[-1], "twin": twin_tokens.shape[-1]},
        "metric": {
            "type": question.get("metric", "logit_diff"),
            "clean": measure(logits),
            "twin": measure(twin_logits),
        },
        "top1": {
            "clean": metrics.top_token(model, logits),
            "twin": metrics.top_token(model, twin_logits),
        },
        "activation": metrics.activation_strength(model, cache),
        "dla": metrics.direct_logit_attribution(model, cache, direction),
        "attention": metrics.attention_from_last(model, cache),
        "top_predictions": metrics.top_predictions(model, logits),
    }


def _task(model, question: dict):
    """For this question's task (protocol §2.3): the function that scores logits, and the
    unembedding direction that DLA projects onto."""
    kind = question.get("metric", "logit_diff")
    if kind == "logit_diff":
        correct = metrics.single_token_id(model, question["correct"])
        wrong = metrics.single_token_id(model, question["wrong"])
        return (
            lambda logits: metrics.logit_diff(logits, correct, wrong),
            metrics.logit_diff_direction(model, correct, wrong),
        )
    if kind == "prob_diff":
        years, year = metrics.year_token_ids(model), question["year"]
        return (
            lambda logits: metrics.greater_than_prob_diff(logits, years, year),
            metrics.greater_than_direction(model, years, year),
        )
    raise KeyError(f"Unknown metric {kind!r} in question {question['id']}")


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
    # Compact (no indentation): result files hold thousands of numbers, and whitespace
    # around each one would roughly triple the file size.
    path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
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
