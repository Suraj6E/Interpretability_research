"""Question sets: generate them from templates, and check results against the
inclusion rules (docs/RESEARCH_PROTOCOL.md §3).

    python -m interp.questions --config configs/questions_v0.yaml

Generation uses no model, so the released set is the same for every model. Each skill
gets its own random generator (seeded from the config seed and the skill name), so
changing one skill's settings does not change the questions of the others.
"""

import argparse
import itertools
import json
import random
from pathlib import Path

import yaml

# Rule 4: the clean-minus-twin gap a question needs, so recovery ratios are stable.
# logit_diff is in logits (protocol §3). prob_diff (greater-than) is in probability
# units; 0.2 is provisional, to be confirmed before the protocol is frozen.
MIN_GAP = {"logit_diff": 1.0, "prob_diff": 0.2}

RULES = ("single_token", "equal_length", "model_correct", "gap", "duplicate_group")


def generate(config: dict) -> list[dict]:
    questions = []
    for skill, settings in config["skills"].items():
        rng = random.Random(f"{config['seed']}-{skill}")
        questions += GENERATORS[skill](settings, rng)
    return questions


def failed_rules(question: dict, result: dict) -> list[str]:
    """Rules 2-4 for one analyzed question (rule 1 fails earlier, while analyzing)."""
    failed = []
    if result["n_tokens"]["clean"] != result["n_tokens"]["twin"]:
        failed.append("equal_length")

    metric = result["metric"]
    top_clean, top_twin = result["top1"]["clean"], result["top1"]["twin"]
    if metric["type"] == "logit_diff":
        correct = (
            metric["clean"] > 0
            and metric["twin"] < 0
            and top_clean == question["correct"]
            and top_twin == question["wrong"]
        )
    else:  # greater-than: the top guess on the clean prompt must be a later year
        correct = (
            metric["clean"] > 0
            and len(top_clean) == 2
            and top_clean.isdigit()
            and int(top_clean) > question["year"]
        )
    if not correct:
        failed.append("model_correct")

    if metric["clean"] - metric["twin"] < MIN_GAP[metric["type"]]:
        failed.append("gap")
    return failed


# ---------- generators, one per skill ----------


def _question(qid, skill, template_id, clean, twin, reference, **answer):
    return {
        "id": qid,
        "skill": skill,
        "template_id": template_id,
        "clean": clean,
        "twin": twin,
        **answer,
        "reference": reference,
    }


def _induction(cfg, rng):
    out = []
    for i in range(cfg["count"]):
        words = rng.sample(cfg["words"], cfg["length"] + 1)
        seq, spare = words[:-1], words[-1]
        k = rng.randint(cfg["min_prefix"], cfg["max_prefix"])  # repeat seq[:k], predict seq[k]
        first_twin = seq[:k] + [spare] + seq[k + 1 :]
        out.append(_question(
            f"ind_{i + 1:03d}", "induction", "ind_t1",
            clean=" ".join(seq + seq[:k]),
            twin=" ".join(first_twin + seq[:k]),
            reference=cfg["reference"],
            metric="logit_diff", correct=" " + seq[k], wrong=" " + spare,
        ))
    return out


def _indirect_object(cfg, rng):
    out, seen = [], set()
    while len(out) < cfg["count"]:
        template = rng.choice(cfg["templates"])
        io, s = rng.sample(cfg["names"], 2)  # io = indirect object (answer), s = subject
        first, second = (io, s) if rng.random() < 0.5 else (s, io)
        fill = {"N1": first, "N2": second,
                "place": rng.choice(cfg["places"]), "object": rng.choice(cfg["objects"])}
        clean = template["text"].format(**fill, S=s)
        if clean in seen:
            continue
        seen.add(clean)
        out.append(_question(
            f"ioi_{len(out) + 1:03d}", "indirect_object", template["id"],
            clean=clean,
            twin=template["text"].format(**fill, S=io),
            reference=cfg["reference"],
            metric="logit_diff", correct=" " + io, wrong=" " + s,
        ))
    return out


def _greater_than(cfg, rng):
    out = []
    years = rng.sample(range(cfg["min_year"], cfg["max_year"] + 1), cfg["count"])
    for i, year in enumerate(years):
        template = rng.choice(cfg["templates"])
        fill = {"noun": rng.choice(cfg["nouns"]), "century": cfg["century"]}
        out.append(_question(
            f"gt_{i + 1:03d}", "greater_than", template["id"],
            clean=template["text"].format(**fill, start=f"{cfg['century']}{year:02d}"),
            twin=template["text"].format(**fill, start=f"{cfg['century']}01"),
            reference=cfg["reference"],
            metric="prob_diff", year=year,
        ))
    return out


def _fact_recall(cfg, rng):
    out, n = [], 0
    for group in cfg["groups"].values():
        facts = list(group["facts"].items())
        for template, (subject, answer) in itertools.product(group["templates"], facts):
            n += 1
            partners = [(s, a) for s, a in facts if a != answer]
            for j, (other, other_answer) in enumerate(
                rng.sample(partners, min(cfg["twins_per_prompt"], len(partners)))
            ):
                out.append(_question(
                    f"fact_{n:03d}_{j + 1}", "fact_recall", template["id"],
                    clean=template["text"].format(subject=subject),
                    twin=template["text"].format(subject=other),
                    reference=cfg["reference"],
                    metric="logit_diff", correct=" " + answer, wrong=" " + other_answer,
                    group=f"fact_{n:03d}",
                ))
    return out


def _agreement(cfg, rng):
    combos = list(itertools.product(
        cfg["templates"], cfg["subjects"], cfg["nouns"], cfg["preps"], (True, False)
    ))
    out = []
    for i, (template, (s_sg, s_pl), (n_sg, n_pl), prep, plural) in enumerate(
        rng.sample(combos, cfg["count"])
    ):
        # The intervening noun has the opposite number to the clean subject.
        if plural:
            clean_s, twin_s, noun, correct, wrong = s_pl, s_sg, n_sg, " are", " is"
        else:
            clean_s, twin_s, noun, correct, wrong = s_sg, s_pl, n_pl, " is", " are"
        out.append(_question(
            f"sva_{i + 1:03d}", "agreement", template["id"],
            clean=template["text"].format(subject=clean_s, prep=prep, noun=noun),
            twin=template["text"].format(subject=twin_s, prep=prep, noun=noun),
            reference=cfg["reference"],
            metric="logit_diff", correct=correct, wrong=wrong,
        ))
    return out


GENERATORS = {
    "induction": _induction,
    "indirect_object": _indirect_object,
    "greater_than": _greater_than,
    "fact_recall": _fact_recall,
    "agreement": _agreement,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="question-set config (YAML)")
    args = parser.parse_args(argv)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    questions = generate(config)
    output = Path(config["output"])
    output.write_text(json.dumps({
        "version": config["version"],
        "seed": config["seed"],
        "generated_by": f"python -m interp.questions --config {args.config.as_posix()}",
        "note": "All generated pairs. Each model uses the subset that passes the inclusion "
                "rules; see results/<model>/filter_report.json.",
        "questions": questions,
    }, indent=1), encoding="utf-8")

    counts = {}
    for q in questions:
        counts[q["skill"]] = counts.get(q["skill"], 0) + 1
    print(f"{len(questions)} questions -> {output.as_posix()}: {counts}")


if __name__ == "__main__":
    main()
