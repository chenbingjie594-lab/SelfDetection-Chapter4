#!/usr/bin/env python3
"""Unified, audit-gated E0/E1/E2 training entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from chapter4_detection.scripts.model_audit import compatible_transfer


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def require_data_gate(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"Formal training blocked by data audit: {report.get('failures')}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("E0", "E1", "E2"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 3407, 2026), required=True)
    parser.add_argument("--common", type=Path, default=Path("chapter4_detection/configs/common.yaml"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Required E2 sliced data.yaml; E0/E1 use common data")
    parser.add_argument("--epochs", type=int, help="Smoke-only override; omit for formal protocol")
    args = parser.parse_args()
    common, experiment = load_yaml(args.common), load_yaml(args.config)
    if experiment["experiment"] != args.experiment:
        raise ValueError("Experiment/config mismatch")
    audit = require_data_gate(Path(common["audit_report"]))
    data = args.data if args.experiment == "E2" else Path(common["data"])
    if args.experiment == "E2" and not args.data:
        raise ValueError("E2 requires --data pointing to the generated slice data.yaml")

    if args.experiment == "E1":
        model = YOLO(experiment["model"])
        transfer = compatible_transfer(model, Path(experiment["pretrained"]))
    else:
        model = YOLO(experiment["model"])
        transfer = None
    name = f"{args.experiment.lower()}_seed{args.seed}" + ("_smoke" if args.epochs else "")
    run_dir = Path(common["project"]) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "experiment": args.experiment,
        "seed": args.seed,
        "git_commit": git_commit(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_manifest_sha256": audit["manifest_sha256"],
        "data_summary": audit["summary"],
        "config": {**common, **experiment},
        "transfer": transfer,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    train_keys = (
        "patience", "batch", "device", "workers", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v",
        "translate", "scale", "fliplr", "mosaic", "mixup", "close_mosaic", "amp", "deterministic",
    )
    kwargs = {key: common[key] for key in train_keys}
    kwargs.update(
        data=str(data), epochs=args.epochs or common["epochs"], imgsz=experiment["imgsz"], seed=args.seed,
        project=str(Path(common["project"])), name=name, exist_ok=False, pretrained=True, val=True, plots=True,
    )
    model.train(**kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
