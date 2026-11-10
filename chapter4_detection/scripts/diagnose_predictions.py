#!/usr/bin/env python3
"""Diagnostic (non-AP) analysis with train-frozen size/contrast thresholds."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from chapter4_detection.scripts.freeze_diagnostic_thresholds import local_contrast
from chapter4_detection.scripts.slicing import box_iou, yolo_to_xyxy


def bin_name(value: float, edges: list[float], labels: tuple[str, str, str]) -> str:
    return labels[0] if value <= edges[0] else labels[1] if value <= edges[1] else labels[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text())["images"]
    thresholds = json.loads(args.thresholds.read_text())["classes"]
    diagnostic = defaultdict(lambda: Counter(total=0, detected_iou50=0))
    class_misses, false_positives, recall_by_iou = Counter(), Counter(), Counter()
    iou_thresholds = (0.5, 0.6, 0.7, 0.75, 0.8, 0.9)
    total_gt = 0
    for name, record in sorted(predictions.items()):
        gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        rows = [line.split() for line in (args.labels / f"{Path(name).stem}.txt").read_text().splitlines() if line.strip()]
        gt_cls = np.asarray([int(row[0]) for row in rows], dtype=int)
        gt_boxes = yolo_to_xyxy(np.asarray([[float(x) for x in row[1:]] for row in rows]), record["width"], record["height"])
        pred = [p for p in record["predictions"] if p["confidence"] >= args.confidence]
        pred_boxes = np.asarray([p["xyxy"] for p in pred], dtype=float).reshape(-1, 4)
        pred_cls = np.asarray([p["class_id"] for p in pred], dtype=int)
        matched_pred = set()
        for cls, box in zip(gt_cls, gt_boxes):
            total_gt += 1
            candidates = np.flatnonzero(pred_cls == cls)
            ious = box_iou(box, pred_boxes[candidates]) if len(candidates) else np.asarray([])
            best_local = int(np.argmax(ious)) if len(ious) else -1
            best_iou = float(ious[best_local]) if len(ious) else 0.0
            if best_local >= 0 and best_iou >= 0.5:
                matched_pred.add(int(candidates[best_local]))
            for threshold in iou_thresholds:
                recall_by_iou[str(threshold)] += int(best_iou >= threshold)
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            contrast = local_contrast(gray, box)
            size_bin = bin_name(area, thresholds[str(cls)]["area_px_tertiles"], ("tiny", "medium", "large"))
            contrast_bin = bin_name(contrast, thresholds[str(cls)]["contrast_tertiles"], ("low", "medium", "high"))
            for key in (f"class{cls}/size/{size_bin}", f"class{cls}/contrast/{contrast_bin}"):
                diagnostic[key]["total"] += 1
                diagnostic[key]["detected_iou50"] += int(best_iou >= 0.5)
            if best_iou < 0.5:
                class_misses[str(cls)] += 1
        for index, cls in enumerate(pred_cls):
            if index not in matched_pred:
                false_positives[str(cls)] += 1
    report = {
        "kind": "diagnostic_not_official_AP",
        "confidence": args.confidence,
        "recall_by_iou": {key: value / max(total_gt, 1) for key, value in recall_by_iou.items()},
        "class_misses": dict(class_misses),
        "background_false_positives": dict(false_positives),
        "groups": {key: {**value, "recall_iou50": value["detected_iou50"] / max(value["total"], 1)} for key, value in diagnostic.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
