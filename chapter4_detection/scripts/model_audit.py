#!/usr/bin/env python3
"""Build E0/E1, report complexity and deterministic compatible weight transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def compatible_transfer(model: YOLO, weights: Path) -> dict:
    source = YOLO(str(weights)).model.state_dict()
    target = model.model.state_dict()
    matched = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    incompatible = sorted(k for k, v in source.items() if k in target and target[k].shape != v.shape)
    unexpected = sorted(k for k in source if k not in target)
    missing = sorted(k for k in target if k not in matched)
    model.model.load_state_dict(matched, strict=False)
    return {
        "matched_tensors": len(matched),
        "matched_parameters": int(sum(v.numel() for v in matched.values())),
        "target_tensors": len(target),
        "target_parameters": int(sum(v.numel() for v in target.values())),
        "missing_target_keys": missing,
        "shape_incompatible_source_keys": incompatible,
        "unexpected_source_keys": unexpected,
        "new_parameter_initialization": "Ultralytics/PyTorch module default initialization retained for unmatched tensors",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = YOLO(args.model)
    transfer = compatible_transfer(model, args.weights) if args.weights else None
    parameters = int(sum(p.numel() for p in model.model.parameters()))
    report = {
        "model": args.model,
        "weights": str(args.weights) if args.weights else None,
        "parameters": parameters,
        "gflops_at_imgsz": float(get_flops(model.model, imgsz=args.imgsz)),
        "imgsz": args.imgsz,
        "strides": [float(x) for x in model.model.stride.cpu()],
        "transfer": transfer,
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
