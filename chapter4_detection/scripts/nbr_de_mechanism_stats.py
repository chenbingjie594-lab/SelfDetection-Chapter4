#!/usr/bin/env python3
"""Measure NBR-DE responses inside GT regions versus normal background."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from chapter4_detection.models.nbr_de import NBRDE, register_nbr_de
from chapter4_detection.scripts.evaluate_original import load_gt


def summarize(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    return {"count": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(np.median(x)), "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75))}


def checkpoint_gamma(path: Path) -> float:
    model = YOLO(str(path))
    module = next(m for m in model.model.modules() if isinstance(m, NBRDE))
    return float(module.gamma.detach().cpu())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--last-weights", type=Path)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=1536)
    a = p.parse_args(); register_nbr_de()
    model = YOLO(str(a.weights))
    image_paths = sorted(x for x in a.images.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"})
    first = cv2.imread(str(image_paths[0]))
    model.predict(first, imgsz=a.imgsz, conf=.001, iou=.7, rect=True, device=a.device, verbose=False)
    module = next(m for m in model.predictor.model.model.modules() if isinstance(m, NBRDE))
    module.export_features = True
    gamma = float(module.gamma.detach().cpu())
    values = {region: {key: [] for key in ("difference_abs", "gate", "contribution_abs", "contribution_signed", "relative_change")}
              for region in ("defect", "background")}
    per_image = []
    for path in image_paths:
        image = cv2.imread(str(path)); h, w = image.shape[:2]
        model.predict(image, imgsz=a.imgsz, conf=.001, iou=.7, rect=True, device=a.device, verbose=False)
        maps = module.last_features
        f, d, g = (maps[k][0].float() for k in ("F", "D", "G"))
        contribution = gamma * g * d
        fh, fw = f.shape[-2:]
        classes, boxes = load_gt(a.labels / f"{path.stem}.txt", w, h)
        mask = torch.zeros((fh, fw), dtype=torch.bool, device=f.device)
        for box in boxes:
            x1, y1, x2, y2 = box.tolist()
            ix1, ix2 = max(0, int(np.floor(x1 * fw / w))), min(fw, int(np.ceil(x2 * fw / w)))
            iy1, iy2 = max(0, int(np.floor(y1 * fh / h))), min(fh, int(np.ceil(y2 * fh / h)))
            mask[iy1:iy2, ix1:ix2] = True
        # Keep all defect cells; deterministically subsample background to avoid domination.
        bg = ~mask
        for region, selected in (("defect", mask), ("background", bg)):
            if not selected.any():
                continue
            metrics = {
                "difference_abs": d.abs().mean(0)[selected], "gate": g.mean(0)[selected],
                "contribution_abs": contribution.abs().mean(0)[selected],
                "contribution_signed": contribution.mean(0)[selected],
                "relative_change": contribution.abs().mean(0)[selected] / (f.abs().mean(0)[selected] + 1e-9),
            }
            for key, tensor in metrics.items():
                array = tensor.detach().cpu().numpy()
                if region == "background" and len(array) > 512:
                    array = array[np.linspace(0, len(array) - 1, 512, dtype=int)]
                values[region][key].extend(array.tolist())
        per_image.append({"image": path.name, "gt_instances": len(classes), "feature_shape": [int(fh), int(fw)]})
    output = {"weights": str(a.weights), "mode": module.mode, "gamma_initial": 0.0, "gamma_best": gamma,
              "gamma_last": checkpoint_gamma(a.last_weights) if a.last_weights else None,
              "gamma_history_note": "Only initialized, best, and final checkpoints exist; per-epoch gamma was not logged.",
              "actual_fusion": "gamma * G * D", "regions": {}, "images": per_image}
    for region in values:
        output["regions"][region] = {key: summarize(vals) for key, vals in values[region].items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: output[k] for k in ("mode", "gamma_initial", "gamma_best", "gamma_last", "regions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
