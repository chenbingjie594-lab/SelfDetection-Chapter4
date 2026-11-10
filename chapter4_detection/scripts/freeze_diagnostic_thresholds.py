#!/usr/bin/env python3
"""Freeze size and local-contrast bins using training GT only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chapter4_detection.scripts.slicing import yolo_to_xyxy


def local_contrast(gray: np.ndarray, box: np.ndarray, ring_scale: float = 1.5) -> float:
    x1, y1, x2, y2 = box.astype(int)
    height, width = gray.shape
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    rx1, ry1 = max(0, int(x1 - bw * (ring_scale - 1))), max(0, int(y1 - bh * (ring_scale - 1)))
    rx2, ry2 = min(width, int(x2 + bw * (ring_scale - 1))), min(height, int(y2 + bh * (ring_scale - 1)))
    defect = gray[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    context = gray[ry1:ry2, rx1:rx2].copy()
    mask = np.ones(context.shape, dtype=bool)
    mask[max(0, y1 - ry1):max(0, y2 - ry1), max(0, x1 - rx1):max(0, x2 - rx1)] = False
    ring = context[mask]
    return float(abs(float(defect.mean()) - float(ring.mean())) / max(float(ring.std()), 1e-6))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = {0: {"area": [], "contrast": []}, 1: {"area": [], "contrast": []}}
    for image_path in sorted(args.images.glob("*")):
        if not image_path.is_file():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        rows = [line.split() for line in (args.labels / f"{image_path.stem}.txt").read_text().splitlines() if line.strip()]
        boxes = yolo_to_xyxy(np.asarray([[float(x) for x in row[1:]] for row in rows]), image.shape[1], image.shape[0])
        for row, box in zip(rows, boxes):
            cls = int(row[0])
            values[cls]["area"].append(float((box[2] - box[0]) * (box[3] - box[1])))
            values[cls]["contrast"].append(local_contrast(image, box))
    report = {"source": "training GT only", "classes": {}}
    for cls, metrics in values.items():
        report["classes"][str(cls)] = {
            "instances": len(metrics["area"]),
            "area_px_tertiles": np.quantile(metrics["area"], [1 / 3, 2 / 3]).tolist(),
            "contrast_tertiles": np.quantile(metrics["contrast"], [1 / 3, 2 / 3]).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
