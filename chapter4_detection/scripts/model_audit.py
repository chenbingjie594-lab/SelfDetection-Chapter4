#!/usr/bin/env python3
"""Build E0/E1, report complexity and deterministic compatible weight transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import re

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def _semantic_target_key(source_key: str) -> tuple[str | None, str]:
    """Map YOLO11s layers to equal-semantic P2 layers (not merely equal names)."""
    match = re.match(r"model\.(\d+)\.(.*)", source_key)
    if not match:
        return None, "not_a_model_layer"
    layer, suffix = int(match.group(1)), match.group(2)
    if layer <= 16:
        return source_key, "unchanged_backbone_or_top_down_neck"
    layer_map = {17: 23, 19: 25, 20: 26, 22: 28}
    if layer in layer_map:
        return f"model.{layer_map[layer]}.{suffix}", "equivalent_bottom_up_neck_layer"
    if layer == 23:
        branch = re.match(r"((?:one2one_)?cv[23])\.(\d+)\.(.*)", suffix)
        if branch:
            family, scale, rest = branch.groups()
            return f"model.29.{family}.{int(scale) + 1}.{rest}", "detect_P3_P4_P5_branch_shifted_after_new_P2"
        if suffix.startswith("dfl."):
            return f"model.29.{suffix}", "shared_DFL_module"
        return None, "incompatible_three_scale_detect_metadata"
    return None, "layer_has_no_equal_semantic_counterpart"


def semantic_transfer(model: YOLO, weights: Path) -> dict:
    source = YOLO(str(weights)).model.state_dict()
    target = model.model.state_dict()
    matched, mapping, rejected = {}, [], []
    for source_key, value in source.items():
        target_key, reason = _semantic_target_key(source_key)
        if target_key is None:
            rejected.append({"source": source_key, "reason": reason})
        elif target_key not in target:
            rejected.append({"source": source_key, "target": target_key, "reason": "target_key_absent"})
        elif target[target_key].shape != value.shape:
            rejected.append({"source": source_key, "target": target_key, "reason": "shape_mismatch",
                             "source_shape": list(value.shape), "target_shape": list(target[target_key].shape)})
        else:
            matched[target_key] = value
            mapping.append({"source": source_key, "target": target_key, "reason": reason, "elements": value.numel()})
    model.model.load_state_dict(matched, strict=False)
    loaded = model.model.state_dict()
    verification_failures = [m for m in mapping if not torch.equal(loaded[m["target"]].cpu(), source[m["source"]].cpu())]
    missing = sorted(k for k in target if k not in matched)
    matched_parameters = int(sum(v.numel() for v in matched.values()))
    target_parameters = int(sum(v.numel() for v in target.values()))
    return {
        "matched_tensors": len(matched),
        "matched_parameters": matched_parameters,
        "matched_parameter_ratio": matched_parameters / target_parameters,
        "target_tensors": len(target),
        "target_parameters": target_parameters,
        "explicit_mapping": mapping,
        "missing_target_keys": missing,
        "unmapped_source": rejected,
        "value_verification_failures": verification_failures,
        "value_verification_passed": not verification_failures,
        "new_parameter_initialization": "Ultralytics/PyTorch module default initialization retained for unmatched tensors",
    }


def semantic_transfer_e3(model: YOLO, weights: Path) -> dict:
    """Transfer YOLO11s tensors across the single NBR-DE insertion at layer 17."""
    source = YOLO(str(weights)).model.state_dict()
    target = model.model.state_dict()
    matched, mapping, rejected = {}, [], []
    for source_key, value in source.items():
        match = re.match(r"model\.(\d+)\.(.*)", source_key)
        if not match:
            rejected.append({"source": source_key, "reason": "not_a_model_layer"})
            continue
        layer, suffix = int(match.group(1)), match.group(2)
        target_layer = layer if layer <= 16 else layer + 1
        target_key = f"model.{target_layer}.{suffix}"
        if target_key not in target or target[target_key].shape != value.shape:
            rejected.append({"source": source_key, "target": target_key, "reason": "absent_or_shape_mismatch"})
            continue
        matched[target_key] = value
        mapping.append({"source": source_key, "target": target_key, "reason": "same_semantic_layer_shifted_after_NBR_DE", "elements": value.numel()})
    model.model.load_state_dict(matched, strict=False)
    loaded = model.model.state_dict()
    failures = [m for m in mapping if not torch.equal(loaded[m["target"]].cpu(), source[m["source"]].cpu())]
    total = sum(v.numel() for v in target.values())
    return {
        "matched_tensors": len(matched), "matched_parameters": sum(v.numel() for v in matched.values()),
        "matched_parameter_ratio": sum(v.numel() for v in matched.values()) / total,
        "target_tensors": len(target), "target_parameters": total, "explicit_mapping": mapping,
        "missing_target_keys": sorted(k for k in target if k not in matched), "unmapped_source": rejected,
        "value_verification_failures": failures, "value_verification_passed": not failures,
        "new_parameter_initialization": "NBR-DE defaults with gamma=0; other unmatched tensors retain defaults",
    }


# Backward-compatible import for earlier scripts/tests; implementation is now semantic.
compatible_transfer = semantic_transfer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = YOLO(args.model)
    transfer = semantic_transfer(model, args.weights) if args.weights else None
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
