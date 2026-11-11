#!/usr/bin/env python3
"""Mutually exclusive miss, localization, group-recall and false-positive diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from chapter4_detection.scripts.freeze_diagnostic_thresholds import local_contrast
from chapter4_detection.scripts.slicing import box_iou, yolo_to_xyxy


NAMES = {0: "Flash point", 1: "Big black spots"}


def bin_name(value: float, edges: list[float], labels: tuple[str, str, str]) -> str:
    return labels[0] if value <= edges[0] else labels[1] if value <= edges[1] else labels[2]


def greedy_matches(gt_cls: np.ndarray, gt_boxes: np.ndarray, pred_cls: np.ndarray, pred_boxes: np.ndarray,
                   pred_conf: np.ndarray, iou_threshold: float, confidence: float) -> list[tuple[int, int, float]]:
    candidates = []
    for gi, (cls, box) in enumerate(zip(gt_cls, gt_boxes)):
        indices = np.flatnonzero((pred_cls == cls) & (pred_conf >= confidence))
        if len(indices):
            for pi, iou in zip(indices, box_iou(box, pred_boxes[indices])):
                if iou >= iou_threshold:
                    candidates.append((float(iou), float(pred_conf[pi]), gi, int(pi)))
    matches, used_gt, used_pred = [], set(), set()
    for iou, conf, gi, pi in sorted(candidates, reverse=True):
        if gi not in used_gt and pi not in used_pred:
            matches.append((gi, pi, iou))
            used_gt.add(gi)
            used_pred.add(pi)
    return matches


def best_candidate(indices: np.ndarray, box: np.ndarray, pred_boxes: np.ndarray,
                   pred_conf: np.ndarray) -> tuple[int | None, float]:
    if not len(indices):
        return None, 0.0
    ious = box_iou(box, pred_boxes[indices])
    order = sorted(range(len(indices)), key=lambda k: (float(ious[k]), float(pred_conf[indices[k]])), reverse=True)
    k = order[0]
    return int(indices[k]), float(ious[k])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.confidence != 0.25:
        raise ValueError("Formal diagnostic confidence is frozen at 0.25")
    prediction_file = json.loads(args.predictions.read_text(encoding="utf-8"))
    predictions = prediction_file["images"]
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))["classes"]
    groups = defaultdict(lambda: Counter(total=0, detected=0))
    misses, errors, localizations = Counter(), [], []
    false_predictions = Counter()
    false_prediction_rows = []
    match_counts = {str(iou): Counter() for iou in (0.5, 0.75, 0.9)}
    totals = Counter()

    for name, record in sorted(predictions.items()):
        gray = cv2.imread(str(args.images / name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Missing image: {name}")
        rows = [line.split() for line in (args.labels / f"{Path(name).stem}.txt").read_text().splitlines() if line.strip()]
        gt_cls = np.asarray([int(row[0]) for row in rows], dtype=int)
        gt_boxes = yolo_to_xyxy(np.asarray([[float(x) for x in row[1:]] for row in rows]), record["width"], record["height"])
        pred = record["predictions"]
        pred_boxes = np.asarray([p["xyxy"] for p in pred], dtype=float).reshape(-1, 4)
        pred_cls = np.asarray([p["class_id"] for p in pred], dtype=int)
        pred_conf = np.asarray([p["confidence"] for p in pred], dtype=float)
        official = greedy_matches(gt_cls, gt_boxes, pred_cls, pred_boxes, pred_conf, 0.5, args.confidence)
        official_by_gt = {gi: (pi, iou) for gi, pi, iou in official}
        official_pred = {pi for _, pi, _ in official}
        reserved = set(official_pred)

        for threshold in match_counts:
            for gi, _, _ in greedy_matches(gt_cls, gt_boxes, pred_cls, pred_boxes, pred_conf, float(threshold), args.confidence):
                match_counts[threshold][str(int(gt_cls[gi]))] += 1
        for cls in gt_cls:
            totals[str(int(cls))] += 1

        for gi, (cls, box) in enumerate(zip(gt_cls, gt_boxes)):
            area = float((box[2] - box[0]) * (box[3] - box[1]))
            contrast = float(local_contrast(gray, box))
            size_group = bin_name(area, thresholds[str(cls)]["area_px_tertiles"], ("small", "medium", "large"))
            contrast_group = bin_name(contrast, thresholds[str(cls)]["contrast_tertiles"], ("low", "medium", "high"))
            detected = gi in official_by_gt
            for key in (f"class{cls}/size/{size_group}", f"class{cls}/contrast/{contrast_group}"):
                groups[key]["total"] += 1
                groups[key]["detected"] += int(detected)
            if detected:
                pi, iou = official_by_gt[gi]
                pw, ph = pred_boxes[pi][2] - pred_boxes[pi][0], pred_boxes[pi][3] - pred_boxes[pi][1]
                gw, gh = box[2] - box[0], box[3] - box[1]
                pc = (pred_boxes[pi][:2] + pred_boxes[pi][2:]) / 2
                gc = (box[:2] + box[2:]) / 2
                delta = pc - gc
                localizations.append({"image": name, "class_id": int(cls), "gt_xyxy": box.tolist(),
                                      "prediction_xyxy": pred_boxes[pi].tolist(), "confidence": float(pred_conf[pi]),
                                      "iou": float(iou), "center_distance_px": float(np.linalg.norm(delta)),
                                      "normalized_center_dx": float(abs(delta[0]) / max(gw, 1e-9)),
                                      "normalized_center_dy": float(abs(delta[1]) / max(gh, 1e-9)),
                                      "relative_width_error": float(abs(pw - gw) / max(gw, 1e-9)),
                                      "relative_height_error": float(abs(ph - gh) / max(gh, 1e-9))})
                continue

            available = np.asarray([i for i in range(len(pred)) if i not in reserved], dtype=int)
            low = available[(pred_cls[available] == cls) & (pred_conf[available] < args.confidence)] if len(available) else available
            low_pi, low_iou = best_candidate(low, box, pred_boxes, pred_conf)
            same = available[pred_cls[available] == cls] if len(available) else available
            loc_pi, loc_iou = best_candidate(same, box, pred_boxes, pred_conf)
            wrong = available[pred_cls[available] != cls] if len(available) else available
            wrong_pi, wrong_iou = best_candidate(wrong, box, pred_boxes, pred_conf)
            if low_pi is not None and low_iou >= 0.5:
                error_type, pi, max_iou = "low_confidence", low_pi, low_iou
            elif loc_pi is not None and 0.1 <= loc_iou < 0.5:
                error_type, pi, max_iou = "localization", loc_pi, loc_iou
            elif wrong_pi is not None and wrong_iou >= 0.5:
                error_type, pi, max_iou = "class_confusion", wrong_pi, wrong_iou
            else:
                error_type, pi, max_iou = "no_effective_response", None, max(low_iou, loc_iou, wrong_iou)
            if pi is not None:
                reserved.add(pi)
            misses[f"class{cls}/{error_type}"] += 1
            errors.append({"image": name, "gt_class_id": int(cls), "gt_class_name": NAMES[int(cls)],
                           "model": prediction_file.get("model"),
                           "gt_xyxy": box.tolist(), "error_type": error_type,
                           "prediction_xyxy": pred_boxes[pi].tolist() if pi is not None else None,
                           "prediction_class_id": int(pred_cls[pi]) if pi is not None else None,
                           "prediction_confidence": float(pred_conf[pi]) if pi is not None else None,
                           "max_iou": float(max_iou), "size_group": size_group,
                           "contrast_group": contrast_group})

        # Mutually exclusive explanation for confidence-filtered predictions left after official matching.
        for pi in np.flatnonzero(pred_conf >= args.confidence):
            if pi in official_pred:
                continue
            all_ious = np.asarray([box_iou(box, pred_boxes[[pi]])[0] for box in gt_boxes]) if len(gt_boxes) else np.asarray([])
            best_gt = int(np.argmax(all_ious)) if len(all_ious) else None
            best_iou = float(all_ious[best_gt]) if best_gt is not None else 0.0
            if best_gt is not None and best_iou >= 0.5 and pred_cls[pi] == gt_cls[best_gt]:
                category = "duplicate"
            elif best_gt is not None and best_iou >= 0.5:
                category = "class_confusion"
            elif best_gt is not None and best_iou >= 0.1:
                category = "insufficient_localization"
            else:
                category = "background"
            false_predictions[f"class{int(pred_cls[pi])}/{category}"] += 1
            false_prediction_rows.append({"image": name, "prediction_class_id": int(pred_cls[pi]),
                                          "model": prediction_file.get("model"),
                                          "prediction_xyxy": pred_boxes[pi].tolist(),
                                          "confidence": float(pred_conf[pi]), "max_gt_iou": best_iou,
                                          "category": category,
                                          "best_gt_class_id": int(gt_cls[best_gt]) if best_gt is not None else None})

    localization_summary = {}
    for cls in (0, 1):
        subset = [row for row in localizations if row["class_id"] == cls]
        localization_summary[str(cls)] = {key: float(np.mean([r[key] for r in subset])) if subset else None for key in
                                          ("center_distance_px", "normalized_center_dx", "normalized_center_dy",
                                           "relative_width_error", "relative_height_error", "iou")}
    report = {
        "kind": "diagnostic_not_official_AP", "model": prediction_file.get("model"),
        "confidence": args.confidence, "error_priority": ["low_confidence", "localization", "class_confusion", "no_effective_response"],
        "matching": "global greedy one-to-one, descending IoU then confidence, class-aware for detections",
        "totals_by_class": dict(totals), "miss_reasons": dict(misses), "errors": errors,
        "match_rates_by_iou": {threshold: {str(cls): match_counts[threshold][str(cls)] / max(totals[str(cls)], 1) for cls in (0, 1)}
                               for threshold in match_counts},
        "localization_summary": localization_summary, "localization_rows": localizations,
        "false_prediction_categories": dict(false_predictions),
        "false_prediction_rows": false_prediction_rows,
        "groups": {key: {**value, "recall_iou50": value["detected"] / max(value["total"], 1)} for key, value in groups.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("errors", "localization_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
