#!/usr/bin/env python3
"""Aggregate frozen Stage-4 metrics and build a four-model GT-instance table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from chapter4_detection.scripts.evaluate_original import load_gt
from chapter4_detection.scripts.freeze_diagnostic_thresholds import local_contrast
from chapter4_detection.scripts.slicing import box_iou


MODELS = {
    "E0": "e0_seed42_validator_predictions.json",
    "A1": "e3_no_reference_seed42_validator_predictions.json",
    "A2": "e3_no_gate_seed42_validator_predictions.json",
    "E3": "e3_seed42_validator_predictions.json",
}


def best_match(preds: list[dict], cls: int, gt_box: np.ndarray) -> tuple[bool, float, float]:
    candidates = [p for p in preds if int(p["class_id"]) == cls]
    if not candidates:
        return False, 0.0, 0.0
    ious = np.asarray([box_iou(np.asarray(p["xyxy"]), gt_box[None])[0] for p in candidates])
    valid = np.flatnonzero(ious >= 0.5)
    if not len(valid):
        index = int(np.argmax(ious))
        return False, float(ious[index]), float(candidates[index]["confidence"])
    index = max(valid, key=lambda i: candidates[int(i)]["confidence"])
    return float(candidates[int(index)]["confidence"]) >= 0.25, float(ious[index]), float(candidates[int(index)]["confidence"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("chapter4_detection/results"))
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    a = p.parse_args()
    predictions = {key: json.loads((a.results / value).read_text())["images"] for key, value in MODELS.items()}
    thresholds = json.loads(a.thresholds.read_text())["classes"]
    fields = ["image", "gt_index", "class_id", "class_name", "area_px", "size_group", "contrast", "contrast_group"]
    for model in MODELS:
        fields += [f"{model}_detected", f"{model}_iou", f"{model}_confidence"]
    rows = []
    first = next(iter(predictions.values()))
    for name, record in sorted(first.items()):
        image = cv2.imread(str(a.images / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Missing image: {name}")
        classes, boxes = load_gt(a.labels / f"{Path(name).stem}.txt", record["width"], record["height"])
        for index, (cls_t, box_t) in enumerate(zip(classes.numpy().astype(int), boxes.numpy())):
            cls = int(cls_t); area = float((box_t[2] - box_t[0]) * (box_t[3] - box_t[1]))
            contrast = float(local_contrast(image, box_t))
            area_cuts = thresholds[str(cls)]["area_px_tertiles"]
            contrast_cuts = thresholds[str(cls)]["contrast_tertiles"]
            group = lambda value, cuts: "low" if value <= cuts[0] else ("medium" if value <= cuts[1] else "high")
            row = {"image": name, "gt_index": index, "class_id": cls,
                   "class_name": "Flash point" if cls == 0 else "Big black spots",
                   "area_px": area, "size_group": group(area, area_cuts),
                   "contrast": contrast, "contrast_group": group(contrast, contrast_cuts)}
            for model, records in predictions.items():
                detected, iou, confidence = best_match(records[name]["predictions"], cls, box_t)
                row.update({f"{model}_detected": int(detected), f"{model}_iou": iou,
                            f"{model}_confidence": confidence})
            rows.append(row)
    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    groups = {}
    for model in MODELS:
        groups[model] = {}
        for subset, predicate in {
            "black_low_contrast": lambda r: r["class_id"] == 1 and r["contrast_group"] == "low",
            "black_medium_contrast": lambda r: r["class_id"] == 1 and r["contrast_group"] == "medium",
            "black_high_contrast": lambda r: r["class_id"] == 1 and r["contrast_group"] == "high",
            "black_small": lambda r: r["class_id"] == 1 and r["size_group"] == "low",
        }.items():
            selected = [r for r in rows if predicate(r)]
            detected = sum(r[f"{model}_detected"] for r in selected)
            groups[model][subset] = {"detected": detected, "total": len(selected),
                                     "recall_at_conf025_iou05": detected / len(selected)}
    metrics = {}
    for path in sorted(a.results.glob("e3_seed*_metrics.json")):
        seed = path.stem.split("seed", 1)[1].split("_", 1)[0]
        metrics[seed] = json.loads(path.read_text())
    output = {"protocol": "GT instance: class-aware IoU>=0.5 and confidence>=0.25; frozen training-set bins",
              "models": MODELS, "groups": groups, "e3_seed_metrics": metrics,
              "instance_csv": str(a.output_csv), "instance_count": len(rows)}
    a.output_json.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
