#!/usr/bin/env python3
"""Threshold and low-confidence diagnostics without changing official AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chapter4_detection.scripts.evaluate_original import load_gt
from chapter4_detection.scripts.slicing import box_iou


def match_image(preds: list[dict], gt_cls: np.ndarray, gt_boxes: np.ndarray, threshold: float, cls: int):
    selected = sorted((p for p in preds if p["class_id"] == cls and p["confidence"] >= threshold),
                      key=lambda p: p["confidence"], reverse=True)
    gt_ids = np.flatnonzero(gt_cls == cls)
    unmatched = set(gt_ids.tolist())
    tp, fp = 0, 0
    for pred in selected:
        candidates = np.array(sorted(unmatched), dtype=int)
        if not len(candidates):
            fp += 1
            continue
        ious = box_iou(np.asarray(pred["xyxy"]), gt_boxes[candidates])
        best = int(np.argmax(ious))
        if ious[best] >= 0.5:
            tp += 1
            unmatched.remove(int(candidates[best]))
        else:
            fp += 1
    return tp, fp, len(unmatched)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.predictions.read_text())
    cache = {}
    for name, record in data["images"].items():
        cls, boxes = load_gt(a.labels / f"{Path(name).stem}.txt", record["width"], record["height"])
        cache[name] = (cls.numpy().astype(int), boxes.numpy())

    thresholds = [0.05, 0.10, 0.25, 0.50]
    fixed = {}
    for threshold in thresholds:
        fixed[str(threshold)] = {}
        for cls in (0, 1):
            tp = fp = fn = 0
            for name, record in data["images"].items():
                values = match_image(record["predictions"], *cache[name], threshold, cls)
                tp += values[0]; fp += values[1]; fn += values[2]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            fixed[str(threshold)][str(cls)] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    curves = {}
    for cls in (0, 1):
        points = []
        for threshold in np.linspace(0, 1, 101):
            tp = fp = fn = 0
            for name, record in data["images"].items():
                x = match_image(record["predictions"], *cache[name], float(threshold), cls)
                tp += x[0]; fp += x[1]; fn += x[2]
            points.append({"confidence": float(threshold), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)})
        curves[str(cls)] = points

    recovered, confidences = [], []
    for name, record in data["images"].items():
        gt_cls, gt_boxes = cache[name]
        for index in np.flatnonzero(gt_cls == 1):
            candidates = [p for p in record["predictions"] if p["class_id"] == 1]
            if not candidates:
                continue
            ious = np.array([box_iou(np.asarray(p["xyxy"]), gt_boxes[index:index + 1])[0] for p in candidates])
            valid = np.flatnonzero(ious >= 0.5)
            if len(valid):
                confidence = max(candidates[i]["confidence"] for i in valid)
                if confidence < 0.25:
                    recovered.append({"image": name, "gt_index": int(index), "max_candidate_confidence": confidence})
                    confidences.append(confidence)
    black_fp_025 = fixed["0.25"]["1"]["fp"]
    output = {
        "diagnostic_only": True, "iou": 0.5, "fixed_thresholds": fixed, "pr_curves": curves,
        "black_low_confidence_iou50": {"count": len(recovered), "confidences": confidences, "targets": recovered},
        "additional_black_false_positives_vs_0.25": {
            str(t): fixed[str(t)]["1"]["fp"] - black_fp_025 for t in (0.05, 0.10)
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
