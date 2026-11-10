#!/usr/bin/env python3
"""Evaluate E0/E1 with fixed Ultralytics validation and export extended metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, batch=args.batch, device=args.device, conf=0.001, iou=0.7, plots=True)
    summary = {
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr), "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map), "map75": float(metrics.box.map75),
        "per_class_map50_95": {str(i): float(v) for i, v in enumerate(metrics.box.maps)},
        "parameters": int(sum(p.numel() for p in model.model.parameters())),
        "gflops_at_imgsz": float(get_flops(model.model, imgsz=args.imgsz)),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "speed_ms_per_image": metrics.speed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
