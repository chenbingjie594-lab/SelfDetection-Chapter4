#!/usr/bin/env python3
"""Paired source-group bootstrap for Stage-8 SFDFM versus matched E0."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from chapter4_detection.scripts.paired_group_bootstrap import (
    image_stats,
    prepare_weighted,
    weighted_evaluate,
)

SEEDS = (42, 3407, 2026)
METRICS = ("map50_95", "ap75", "flash_ap50_95", "black_ap50_95")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--sfdfm-dir", type=Path, required=True)
    p.add_argument("--candidate-prefix", default="e5_sfdfm")
    p.add_argument("--candidate-name", default="SFDFM")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261117)
    a = p.parse_args()

    records = json.loads(a.manifest.read_text())["records"]
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record["split"] == "val":
            groups[record["source_group"]].append(record["image_name"])
    group_names = sorted(groups)
    image_group = {name: group for group, names in groups.items() for name in names}
    prepared = {}
    for model, directory, prefix in (
        ("Matched-E0", a.baseline_dir, "matched_e0"),
        (a.candidate_name, a.sfdfm_dir, a.candidate_prefix),
    ):
        for seed in SEEDS:
            payload = json.loads((directory / f"{prefix}_seed{seed}_predictions.json").read_text())
            per_image = {
                name: image_stats(record, a.labels / f"{Path(name).stem}.txt")
                for name, record in payload["images"].items()
            }
            prepared[model, seed] = prepare_weighted(per_image, image_group)

    rng = np.random.default_rng(a.seed)
    distributions = {str(seed): [] for seed in SEEDS}
    distributions["seed_mean"] = []
    for _ in range(a.repeats):
        sample = rng.choice(group_names, size=len(group_names), replace=True)
        multiplicity = {g: int(np.sum(sample == g)) for g in set(sample)}
        differences = []
        for seed in SEEDS:
            delta = weighted_evaluate(prepared[a.candidate_name, seed], multiplicity) - weighted_evaluate(
                prepared["Matched-E0", seed], multiplicity
            )
            distributions[str(seed)].append(delta)
            differences.append(delta)
        distributions["seed_mean"].append(np.mean(differences, axis=0))

    intervals = {}
    for seed, values in distributions.items():
        array = np.asarray(values)
        intervals[seed] = {
            metric: {
                "mean": float(array[:, i].mean()),
                "ci95": [float(x) for x in np.quantile(array[:, i], [0.025, 0.975])],
                "p_gt_0": float(np.mean(array[:, i] > 0)),
            }
            for i, metric in enumerate(METRICS)
        }
    result = {
        "protocol": {
            "comparison": f"{a.candidate_name} - Matched-E0",
            "unit": "source_group",
            "groups": len(group_names),
            "images": sum(map(len, groups.values())),
            "repeats": a.repeats,
            "seed": a.seed,
        },
        "intervals": intervals,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
