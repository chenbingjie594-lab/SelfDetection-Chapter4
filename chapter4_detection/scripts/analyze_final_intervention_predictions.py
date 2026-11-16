#!/usr/bin/env python3
"""Quantify final prediction changes for Stage-6 Full/bypass interventions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-12)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "max": float(np.max(values)) if values else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in (42, 3407, 2026):
        prefix = f"e3_seed{seed}"
        full = json.loads((args.input_dir / f"{prefix}_full_predictions.json").read_text())
        for mode in ("gate_bypass", "residual_bypass"):
            bypass = json.loads((args.input_dir / f"{prefix}_{mode}_predictions.json").read_text())
            for image, payload in full["images"].items():
                fp = [p for p in payload["predictions"] if p["confidence"] >= args.confidence]
                bp = [p for p in bypass["images"][image]["predictions"] if p["confidence"] >= args.confidence]
                available = set(range(len(bp)))
                matched_iou, conf_delta, box_delta = [], [], []
                for pred in sorted(fp, key=lambda p: p["confidence"], reverse=True):
                    candidates = [(iou(pred["xyxy"], bp[j]["xyxy"]), j) for j in available if bp[j]["class_id"] == pred["class_id"]]
                    if not candidates:
                        continue
                    overlap, j = max(candidates)
                    if overlap < 0.5:
                        continue
                    available.remove(j)
                    matched_iou.append(overlap)
                    conf_delta.append(abs(pred["confidence"] - bp[j]["confidence"]))
                    box_delta.append(float(np.mean(np.abs(np.asarray(pred["xyxy"]) - np.asarray(bp[j]["xyxy"])))))
                rows.append({
                    "seed": seed, "image": image, "bypass": mode,
                    "full_detections": len(fp), "bypass_detections": len(bp),
                    "matched_detections": len(matched_iou),
                    "unmatched_full": len(fp) - len(matched_iou),
                    "unmatched_bypass": len(bp) - len(matched_iou),
                    "matched_box_iou_mean": float(np.mean(matched_iou)) if matched_iou else None,
                    "confidence_abs_delta_mean": float(np.mean(conf_delta)) if conf_delta else None,
                    "box_coordinate_abs_delta_px_mean": float(np.mean(box_delta)) if box_delta else None,
                })
    csv_path = args.output_dir / "final_prediction_intervention_per_image.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for mode in ("gate_bypass", "residual_bypass"):
        selected = [r for r in rows if r["bypass"] == mode]
        summary[mode] = {
            "confidence_threshold": args.confidence,
            "images": len(selected),
            "full_detections": sum(r["full_detections"] for r in selected),
            "bypass_detections": sum(r["bypass_detections"] for r in selected),
            "matched_detections": sum(r["matched_detections"] for r in selected),
            "unmatched_full": sum(r["unmatched_full"] for r in selected),
            "unmatched_bypass": sum(r["unmatched_bypass"] for r in selected),
            "matched_box_iou": summarize([r["matched_box_iou_mean"] for r in selected if r["matched_box_iou_mean"] is not None]),
            "confidence_abs_delta": summarize([r["confidence_abs_delta_mean"] for r in selected if r["confidence_abs_delta_mean"] is not None]),
            "box_coordinate_abs_delta_px": summarize([r["box_coordinate_abs_delta_px_mean"] for r in selected if r["box_coordinate_abs_delta_px_mean"] is not None]),
        }
    (args.output_dir / "final_prediction_intervention_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
