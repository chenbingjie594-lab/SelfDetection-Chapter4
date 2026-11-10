#!/usr/bin/env python3
"""Evaluate restored predictions against original-image GT using Ultralytics AP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.metrics import DetMetrics, box_iou

from chapter4_detection.scripts.slicing import yolo_to_xyxy


IOU_VECTOR = torch.linspace(0.5, 0.95, 10)


def match_predictions(pred_cls: torch.Tensor, true_cls: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
    correct = np.zeros((pred_cls.shape[0], IOU_VECTOR.numel()), dtype=bool)
    correct_class = true_cls[:, None] == pred_cls
    iou = iou * correct_class
    for index, threshold in enumerate(IOU_VECTOR.tolist()):
        matches = torch.nonzero(iou >= threshold, as_tuple=False).cpu().numpy()
        if matches.shape[0]:
            matches = matches[np.argsort(iou[matches[:, 0], matches[:, 1]].cpu().numpy())[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), index] = True
    return torch.tensor(correct, dtype=torch.bool)


def load_gt(label_path: Path, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [line.split() for line in label_path.read_text().splitlines() if line.strip()]
    classes = torch.tensor([int(row[0]) for row in rows], dtype=torch.float32)
    boxes = torch.tensor(yolo_to_xyxy(np.asarray([[float(x) for x in row[1:]] for row in rows]), width, height), dtype=torch.float32)
    return classes, boxes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.predictions.read_text())
    metrics = DetMetrics(names={0: "Flash point", 1: "Big black spots"})
    size_rows = []
    for image_index, (name, record) in enumerate(sorted(data["images"].items())):
        label_path = args.labels / f"{Path(name).stem}.txt"
        target_cls, target_boxes = load_gt(label_path, record["width"], record["height"])
        preds = record["predictions"]
        pred_cls = torch.tensor([p["class_id"] for p in preds], dtype=torch.float32)
        pred_boxes = torch.tensor([p["xyxy"] for p in preds], dtype=torch.float32).reshape(-1, 4)
        conf = torch.tensor([p["confidence"] for p in preds], dtype=torch.float32)
        correct = match_predictions(pred_cls, target_cls, box_iou(target_boxes, pred_boxes)) if len(preds) and len(target_boxes) else torch.zeros((len(preds), 10), dtype=torch.bool)
        metrics.stats["tp"].append(correct.cpu().numpy())
        metrics.stats["conf"].append(conf.cpu().numpy())
        metrics.stats["pred_cls"].append(pred_cls.cpu().numpy())
        metrics.stats["target_cls"].append(target_cls.cpu().numpy())
        metrics.stats["target_img"].append(np.full(len(target_cls), image_index))
        areas = ((target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])).numpy()
        size_rows.extend({"image": name, "class_id": int(c), "area_px": float(a)} for c, a in zip(target_cls, areas))
    metrics.process(save_dir=args.output.parent, plot=False)
    p, r, map50, map5095 = metrics.mean_results()
    summary = {
        "precision": p,
        "recall": r,
        "map50": map50,
        "map50_95": map5095,
        "map75": metrics.box.map75,
        "per_class_map50_95": {str(cls): float(metrics.maps[cls]) for cls in range(2)},
        "mean_seconds_per_original": data.get("mean_seconds_per_original"),
        "size_diagnostic_rows": size_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "size_diagnostic_rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
