#!/usr/bin/env python3
"""Build the pre-registered seed42 comparison tables from actual result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(name: str, metrics: dict, predictions: dict, epochs: int | None, updates: int | None) -> dict:
    per = metrics["per_class"]
    gflops = metrics.get("gflops_at_imgsz")
    if gflops is None:
        gflops = metrics.get("gflops_per_tile_at_1024", 0) * 5
    return {
        "model": name, "precision": metrics["precision"], "recall": metrics["recall"],
        "map50": metrics["map50"], "map50_95": metrics["map50_95"], "ap75": metrics["map75"],
        "flash_recall": per["0"]["recall"], "black_recall": per["1"]["recall"],
        "flash_ap50": per["0"]["ap50"], "flash_ap50_95": per["0"]["ap50_95"],
        "black_ap50": per["1"]["ap50"], "black_ap50_95": per["1"]["ap50_95"],
        "parameters": metrics.get("parameters"), "gflops_per_original": gflops,
        "forward_ms_per_original": predictions["mean_forward_seconds_per_original"] * 1000,
        "end_to_end_ms_per_original": predictions["mean_seconds_per_original"] * 1000,
        "inference_peak_gpu_bytes": predictions.get("peak_gpu_memory_bytes"),
        "mean_predictions_per_original": predictions["mean_predictions_per_original"],
        "completed_epochs": epochs, "optimizer_updates": updates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--diagnostics", type=Path, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--updates", type=int, nargs="+", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--error-json", type=Path, required=True)
    parser.add_argument("--error-md", type=Path, required=True)
    args = parser.parse_args()
    lengths = {len(args.models), len(args.metrics), len(args.predictions), len(args.diagnostics), len(args.epochs), len(args.updates)}
    if len(lengths) != 1:
        raise ValueError("All model artifact lists must have equal length")
    metrics, predictions, diagnostics = map(lambda paths: [load(p) for p in paths],
                                            (args.metrics, args.predictions, args.diagnostics))
    rows = [metric_row(*values) for values in zip(args.models, metrics, predictions, args.epochs, args.updates)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# E0/E1/E2 seed42 comparison", "", "## Overall", "",
             "| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash R | Black R |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['map50']:.4f} | {row['map50_95']:.4f} | {row['ap75']:.4f} | {row['flash_recall']:.4f} | {row['black_recall']:.4f} |")
    lines += ["", "## Per-class AP", "", "| Model | Flash AP50 | Flash AP50-95 | Black AP50 | Black AP50-95 |",
              "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['flash_ap50']:.4f} | {row['flash_ap50_95']:.4f} | {row['black_ap50']:.4f} | {row['black_ap50_95']:.4f} |")
    lines += ["", "## Compute", "", "| Model | Params | GFLOPs/original | Forward ms | End-to-end ms | Infer peak bytes | Epochs | Optimizer updates |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['parameters']} | {row['gflops_per_original']:.2f} | {row['forward_ms_per_original']:.2f} | {row['end_to_end_ms_per_original']:.2f} | {row['inference_peak_gpu_bytes']} | {row['completed_epochs']} | {row['optimizer_updates']} |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    error_payload = {name: diag for name, diag in zip(args.models, diagnostics)}
    args.error_json.write_text(json.dumps(error_payload, indent=2) + "\n", encoding="utf-8")
    error_lines = ["# Detection error analysis", "", "Diagnostic only: confidence 0.25; official AP is reported separately.", "",
                   "| Model | Class | Low conf | Localization | Confusion | No response |",
                   "|---|---|---:|---:|---:|---:|"]
    for name, diag in zip(args.models, diagnostics):
        misses = diag["miss_reasons"]
        for cls, label in ((0, "Flash"), (1, "Black")):
            error_lines.append(f"| {name} | {label} | {misses.get(f'class{cls}/low_confidence', 0)} | {misses.get(f'class{cls}/localization', 0)} | {misses.get(f'class{cls}/class_confusion', 0)} | {misses.get(f'class{cls}/no_effective_response', 0)} |")
    args.error_md.write_text("\n".join(error_lines) + "\n", encoding="utf-8")
    print(json.dumps({"models": args.models, "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
