"""Batch mode: run every question in a question file and save one result file per question.

    python -m interp.run --config configs/smoke_gpt2.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from interp.model import load_model, model_spec
from interp.results import analyze_question, run_metadata, save_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="experiment config (YAML)")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    spec = model_spec(config["model"])
    questions_path = Path(config["questions"])
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    out_dir = Path(config["output_dir"]) / spec.key

    model = load_model(spec)
    meta = run_metadata(spec, args.config, questions_path, device=str(model.cfg.device))
    # ASCII only: the default Windows console cannot print arrows or dots.
    print(f"{spec.key} on {meta['device']}: {len(questions)} questions -> {out_dir.as_posix()}")

    for question in questions:
        result = analyze_question(model, question, spec.key)
        result["meta"] = meta
        save_result(result, out_dir)
        ld = result["logit_diff"]
        n = result["n_tokens"]
        warning = "  [clean and twin differ in token length]" if n["clean"] != n["twin"] else ""
        print(f"  {question['id']}: LD clean {ld['clean']:+.2f}, twin {ld['twin']:+.2f}{warning}")


if __name__ == "__main__":
    main()
