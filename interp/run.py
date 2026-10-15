"""Batch mode: run every question in a question file, apply the inclusion rules
(docs/RESEARCH_PROTOCOL.md §3), and save one result file per kept question.

    python -m interp.run --config configs/gpt2_v0.yaml

Writes to <output_dir>/<model>/:
  <id>.json           one file per kept question (the viewer's format)
  index.json          the kept questions, for the viewer's question picker
  filter_report.json  every question with the rules it failed, and counts per skill
"""

import argparse
import json
from pathlib import Path

import yaml

from interp.metrics import AnswerTokenError
from interp.model import load_model, model_spec
from interp.questions import MIN_GAP, RULES, failed_rules
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
    _remove_previous_results(out_dir)

    report, kept_groups = [], set()
    for question in questions:
        # Some questions are twin candidates for the same clean prompt ("group"); only the
        # first one that passes every rule is kept, so later candidates are not analyzed.
        group = question.get("group")
        if group in kept_groups:
            result, failed = None, ["duplicate_group"]
        else:
            try:
                result = analyze_question(model, question, spec.key)
                failed = failed_rules(question, result)
            except AnswerTokenError:
                result, failed = None, ["single_token"]
            if not failed and group:
                kept_groups.add(group)
        if not failed:
            result["meta"] = meta
            save_result(result, out_dir)
        report.append({
            "id": question["id"],
            "skill": question["skill"],
            "kept": not failed,
            "failed": failed,
            "metric": result and result["metric"],
            "top1": result and result["top1"],
        })

    summary = _summarize(report)
    _write_json(out_dir / "filter_report.json", {
        "meta": meta, "min_gap": MIN_GAP, "summary": summary, "questions": report,
    })
    by_id = {q["id"]: q for q in questions}
    _write_json(out_dir / "index.json", {
        "model": spec.key,
        "questions": [
            {"id": r["id"], "skill": r["skill"], "clean": by_id[r["id"]]["clean"]}
            for r in report if r["kept"]
        ],
    })
    _register_model(out_dir.parent, spec.key)
    _print_summary(summary)


def _summarize(report: list[dict]) -> dict:
    """Per skill: questions generated, kept, and how many failed each rule.
    A question can fail several rules, so rule counts can add up to more than the exclusions."""
    summary = {}
    for row in report:
        s = summary.setdefault(row["skill"], {"generated": 0, "kept": 0, **{r: 0 for r in RULES}})
        s["generated"] += 1
        s["kept"] += row["kept"]
        for rule in row["failed"]:
            s[rule] += 1
    return summary


def _print_summary(summary: dict):
    columns = ("generated", "kept") + RULES
    print(f"{'skill':<16}" + "".join(f"{c:>16}" for c in columns))
    for skill, counts in summary.items():
        print(f"{skill:<16}" + "".join(f"{counts[c]:>16}" for c in columns))


def _remove_previous_results(out_dir: Path):
    """Delete result files written by the previous run (listed in its index.json), so
    questions excluded this time do not linger. Other files are left alone."""
    index = out_dir / "index.json"
    if not index.exists():
        return
    for entry in json.loads(index.read_text(encoding="utf-8"))["questions"]:
        (out_dir / f"{entry['id']}.json").unlink(missing_ok=True)


def _register_model(results_dir: Path, model_key: str):
    """Keep results/index.json listing the models that have results (used by the viewer)."""
    path = results_dir / "index.json"
    models = json.loads(path.read_text(encoding="utf-8"))["models"] if path.exists() else []
    if model_key not in models:
        models.append(model_key)
    _write_json(path, {"models": sorted(models)})


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
