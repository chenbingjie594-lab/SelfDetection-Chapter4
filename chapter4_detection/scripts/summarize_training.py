#!/usr/bin/env python3
"""Summarize a completed Ultralytics run, including deterministic optimizer-step accounting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_optimizer_updates(samples: int, epochs: int, batch: int, nominal_batch: int, warmup_epochs: float) -> int:
    """Mirror Ultralytics' ni-last_opt_step >= accumulate schedule (optimizer-step attempts)."""
    batches = math.ceil(samples / batch)
    warmup_iterations = max(round(warmup_epochs * batches), 100) if warmup_epochs > 0 else -1
    accumulate, last_step, updates = max(round(nominal_batch / batch), 1), -1, 0
    for ni in range(epochs * batches):
        if ni <= warmup_iterations:
            accumulate = max(1, round(float(np.interp(ni, [0, warmup_iterations], [1, nominal_batch / batch]))))
        if ni - last_step >= accumulate:
            updates += 1
            last_step = ni
    return updates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curve-output", type=Path, help="Optional auditable copy of the per-epoch results CSV")
    args = parser.parse_args()
    rows = list(csv.DictReader((args.run / "results.csv").open()))
    if not rows:
        raise RuntimeError("No completed epochs")
    metric_key = "metrics/mAP50-95(B)"
    best_index, best = max(enumerate(rows, 1), key=lambda item: float(item[1][metric_key]))
    config = yaml.safe_load((args.run / "args.yaml").read_text())
    log = args.log.read_text(errors="replace")
    gpu_values = [float(value) for line in log.splitlines() if re.search(r"\d+/\d+", line)
                  for value in re.findall(r"(\d+(?:\.\d+)?)G", line)]
    weights = args.run / "weights"
    output = {
        "completed_epochs": len(rows), "best_epoch": best_index,
        "best_epoch_metrics": {key.strip(): float(value) for key, value in best.items() if key.strip() != "epoch"},
        "optimizer_update_attempts": expected_optimizer_updates(args.train_samples, len(rows), int(config["batch"]),
                                                                 int(config.get("nbs", 64)), float(config["warmup_epochs"])),
        "optimizer_update_accounting": "exact protocol schedule; assumes no AMP scaler step was skipped",
        "peak_training_gpu_memory_gib_from_log": max(gpu_values) if gpu_values else None,
        "best_checkpoint_sha256": sha256(weights / "best.pt") if (weights / "best.pt").exists() else None,
        "last_checkpoint_sha256": sha256(weights / "last.pt") if (weights / "last.pt").exists() else None,
        "args": config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if args.curve_output:
        args.curve_output.parent.mkdir(parents=True, exist_ok=True)
        args.curve_output.write_text((args.run / "results.csv").read_text(), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "args"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
