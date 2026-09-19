"""Recompute output-blind stability reports for direct logits and reranker."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import evaluate


def read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed(rows):
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate IDs")
    return result


def chosen(row):
    return row["option_ids"][max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)]


def distribution(row):
    return dict(zip(row["option_ids"], row["probabilities"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--perturbations", type=Path, required=True)
    parser.add_argument("--direct-base", type=Path, required=True)
    parser.add_argument("--direct-perturbations", type=Path, required=True)
    parser.add_argument("--reranker-base", type=Path, required=True)
    parser.add_argument("--reranker-perturbations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold, perturb_gold = read(args.gold), read(args.perturbations)
    original_ids = {row["id"] for row in gold if row["provenance"]["variant"] == "original"}
    missing_gold = [row for row in gold if row["provenance"]["variant"] == "missing"]
    if len(original_ids) != 36 or len(missing_gold) != 36 or len(perturb_gold) != 108:
        raise ValueError("Unexpected stability populations")
    systems = {
        "direct_logits": (args.direct_base, args.direct_perturbations),
        "reranker": (args.reranker_base, args.reranker_perturbations),
    }
    result = {
        "meaning": "Stability under output-blind project-authored variants; accuracy against unchanged labels plus semantic-ID-aligned probability movement.",
        "systems": {},
    }
    variants = defaultdict(list)
    for row in perturb_gold:
        variants[row["provenance"]["variant"]].append(row)
    for name, (base_path, perturb_path) in systems.items():
        base, perturb = indexed(read(base_path)), indexed(read(perturb_path))
        if set(perturb) != {row["id"] for row in perturb_gold} or not original_ids <= set(base):
            raise ValueError(f"Incomplete {name} stability outputs")
        report = {
            "base_original": evaluate.evaluate(
                [row for row in gold if row["id"] in original_ids],
                [base[row["id"]] for row in gold if row["id"] in original_ids],
            ),
            "variants": {},
        }
        for variant, variant_gold in variants.items():
            shifts, flips = [], []
            for row in variant_gold:
                candidate = perturb[row["id"]]
                reference = base[row["provenance"]["base_id"]]
                left, right = distribution(reference), distribution(candidate)
                if set(left) != set(right):
                    raise ValueError("Semantic option IDs changed")
                shifts.append(max(abs(left[key] - right[key]) for key in left))
                if chosen(reference) != chosen(candidate):
                    flips.append({"base_id": row["provenance"]["base_id"], "variant_id": row["id"],
                                  "from": chosen(reference), "to": chosen(candidate)})
            report["variants"][variant] = {
                "evaluation": evaluate.evaluate(variant_gold, [perturb[row["id"]] for row in variant_gold]),
                "argmax_flips": len(flips),
                "flip_rows": flips,
                "mean_max_probability_movement": statistics.mean(shifts),
                "max_probability_movement": max(shifts),
            }
        missing_predictions = [base[row["id"]] for row in missing_gold]
        aligned = evaluate.align(missing_gold, missing_predictions)
        report["missing_evidence"] = {
            "evaluation": evaluate.evaluate(missing_gold, missing_predictions),
            "confident_non_insufficient_at_0_8": sum(
                row["predicted_id"] != "insufficient" and row["confidence"] is not None and row["confidence"] >= .8
                for row in aligned
            ),
        }
        result["systems"][name] = report
    with args.output.open("x") as destination:
        destination.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
