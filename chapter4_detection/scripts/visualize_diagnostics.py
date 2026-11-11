#!/usr/bin/env python3
"""Create deterministic qualitative comparisons from exported predictions and diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chapter4_detection.scripts.slicing import box_iou, yolo_to_xyxy


COLORS = {0: (0, 180, 255), 1: (255, 80, 80)}


def gt_for(labels: Path, name: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in (labels / f"{Path(name).stem}.txt").read_text().splitlines() if line.strip()]
    cls = np.asarray([int(row[0]) for row in rows], dtype=int)
    boxes = yolo_to_xyxy(np.asarray([[float(v) for v in row[1:]] for row in rows]), width, height)
    return cls, boxes


def draw(image: np.ndarray, model: str, gt_cls: np.ndarray, gt_boxes: np.ndarray, predictions: list[dict], conf: float = 0.25) -> np.ndarray:
    canvas = image.copy()
    for cls, box in zip(gt_cls, gt_boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 3)
        cv2.putText(canvas, f"GT c{cls}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60), 2)
    for pred in predictions:
        if pred["confidence"] < conf:
            continue
        box, cls = np.asarray(pred["xyxy"]), int(pred["class_id"])
        iou = max((float(box_iou(gt, box.reshape(1, 4))[0]) for gt in gt_boxes), default=0.0)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS[cls], 2)
        cv2.putText(canvas, f"P c{cls} {pred['confidence']:.2f} IoU {iou:.2f}",
                    (x1, min(y2 + 18, canvas.shape[0] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls], 2)
    cv2.rectangle(canvas, (0, 0), (520, 34), (0, 0, 0), -1)
    cv2.putText(canvas, model, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def crop_around(image: np.ndarray, box: list[float], margin: int = 100) -> tuple[np.ndarray, tuple[int, int]]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    span = max(x2 - x1, y2 - y1, 40) + 2 * margin
    left, top = max(0, int(cx - span / 2)), max(0, int(cy - span / 2))
    right, bottom = min(image.shape[1], int(cx + span / 2)), min(image.shape[0], int(cy + span / 2))
    return image[top:bottom, left:right].copy(), (left, top)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--diagnostics", type=Path, nargs="+", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-comparisons", type=int, default=8)
    args = parser.parse_args()
    if len(args.predictions) != len(args.diagnostics):
        raise ValueError("predictions and diagnostics counts differ")
    preds = [json.loads(p.read_text()) for p in args.predictions]
    diags = [json.loads(p.read_text()) for p in args.diagnostics]
    args.output.mkdir(parents=True, exist_ok=True)

    error_maps = [{row["image"]: row["error_type"] for row in diag["errors"]} for diag in diags]
    all_names = sorted(set().union(*(set(m) for m in error_maps)))
    differing = [name for name in all_names if len({m.get(name, "detected") for m in error_maps}) > 1]
    selected = (differing + [n for n in all_names if n not in differing])[:args.max_comparisons]
    created = []
    for name in selected:
        image = cv2.imread(str(args.images / name))
        record = preds[0]["images"][name]
        gt_cls, gt_boxes = gt_for(args.labels, name, record["width"], record["height"])
        panels = [cv2.resize(draw(image, data["model"], gt_cls, gt_boxes, data["images"][name]["predictions"]), (1024, 256)) for data in preds]
        output = args.output / f"comparison_{Path(name).stem}.jpg"
        cv2.imwrite(str(output), np.vstack(panels))
        created.append(str(output))

    # Fixed rule: first filename/GT order for each requested error family, not hand-picked quality.
    families = ((1, None, "black_miss"), (0, "localization", "flash_localization"))
    for cls, kind, label in families:
        rows = sorted((r for diag in diags for r in diag["errors"]
                       if r["gt_class_id"] == cls and (kind is None or r["error_type"] == kind)),
                      key=lambda r: (r["image"], r["gt_xyxy"]))[:4]
        for index, row in enumerate(rows):
            image = cv2.imread(str(args.images / row["image"]))
            crop, (ox, oy) = crop_around(image, row["gt_xyxy"])
            gt = np.asarray(row["gt_xyxy"]) - np.asarray([ox, oy, ox, oy])
            x1, y1, x2, y2 = map(int, gt)
            cv2.rectangle(crop, (x1, y1), (x2, y2), (60, 220, 60), 2)
            if row["prediction_xyxy"] is not None:
                prediction = np.asarray(row["prediction_xyxy"]) - np.asarray([ox, oy, ox, oy])
                px1, py1, px2, py2 = map(int, prediction)
                cv2.rectangle(crop, (px1, py1), (px2, py2), COLORS[row["prediction_class_id"]], 2)
            text = f"{row.get('model')} {label} {row['error_type']} IoU={row['max_iou']:.2f} conf={row.get('prediction_confidence') or 0:.2f}"
            cv2.putText(crop, text, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            output = args.output / f"{label}_{index}_{Path(row['image']).stem}.jpg"
            cv2.imwrite(str(output), crop)
            created.append(str(output))

    background = sorted((r for diag in diags for r in diag["false_prediction_rows"] if r["category"] == "background"),
                        key=lambda r: (r["image"], -r["confidence"]))[:4]
    for index, row in enumerate(background):
        image = cv2.imread(str(args.images / row["image"]))
        crop, (ox, oy) = crop_around(image, row["prediction_xyxy"])
        box = np.asarray(row["prediction_xyxy"]) - np.asarray([ox, oy, ox, oy])
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(crop, (x1, y1), (x2, y2), COLORS[row["prediction_class_id"]], 2)
        cv2.putText(crop, f"{row.get('model')} background FP conf={row['confidence']:.2f}", (4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        output = args.output / f"background_fp_{index}_{Path(row['image']).stem}.jpg"
        cv2.imwrite(str(output), crop)
        created.append(str(output))
    (args.output / "manifest.json").write_text(json.dumps({"selection_rule": "lexicographic fixed error rules", "files": created}, indent=2) + "\n")
    print(json.dumps({"created": len(created), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
