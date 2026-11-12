#!/usr/bin/env python3
"""Export aligned NBR-DE F/B/D/G response maps with original-image GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from chapter4_detection.models.nbr_de import NBRDE, register_nbr_de
from chapter4_detection.scripts.evaluate_original import load_gt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    register_nbr_de()
    model = YOLO(str(a.weights))
    modules = [m for m in model.model.modules() if isinstance(m, NBRDE)]
    if len(modules) != 1:
        raise RuntimeError(f"Expected one NBRDE, found {len(modules)}")
    module = modules[0]
    a.output.mkdir(parents=True, exist_ok=True)
    for name in a.names:
        path = a.images / name
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unreadable image: {path}")
        h, w = image.shape[:2]
        display_w = min(w, 2048)
        display_h = round(h * display_w / w)
        shown = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA)
        sx, sy = display_w / w, display_h / h
        # The predictor may build a fused runtime copy on first use. Initialize it,
        # then enable capture on the module that actually executes.
        model.predict(image, imgsz=1536, conf=0.001, iou=0.7, rect=True, device=a.device, verbose=False)
        runtime_modules = [m for m in model.predictor.model.model.modules() if isinstance(m, NBRDE)]
        if len(runtime_modules) != 1:
            raise RuntimeError(f"Expected one runtime NBRDE, found {len(runtime_modules)}")
        module = runtime_modules[0]
        module.export_features = True
        model.predict(image, imgsz=1536, conf=0.001, iou=0.7, rect=True, device=a.device, verbose=False)
        maps = {}
        for key, value in module.last_features.items():
            response = value[0].float().mean(0).cpu().numpy() if key == "G" else value[0].float().abs().mean(0).cpu().numpy()
            maps[key] = cv2.resize(response, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
        common_min = min(maps[k].min() for k in ("F", "B", "D"))
        common_max = max(maps[k].max() for k in ("F", "B", "D"))
        gt_cls, gt_boxes = load_gt(a.labels / f"{path.stem}.txt", w, h)
        fig, axes = plt.subplots(1, 5, figsize=(20, 3.5), constrained_layout=True)
        axes[0].imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original + GT")
        for axis, key in zip(axes[1:], ("F", "B", "D", "G")):
            limits = (0, 1) if key == "G" else (common_min, common_max)
            axis.imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB), alpha=0.35)
            axis.imshow(maps[key], cmap="turbo", alpha=0.65, vmin=limits[0], vmax=limits[1])
            axis.set_title(key)
        for axis in axes:
            for cls, box in zip(gt_cls.tolist(), gt_boxes.tolist()):
                x1, y1, x2, y2 = box
                x1, x2, y1, y2 = x1*sx, x2*sx, y1*sy, y2*sy
                axis.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color="lime" if int(cls)==0 else "cyan", linewidth=2))
            axis.axis("off")
        fig.savefig(a.output / f"{path.stem}_nbr_de.png", dpi=120)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
