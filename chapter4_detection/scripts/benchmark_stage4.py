#!/usr/bin/env python3
"""Synchronized, same-GPU Stage-4 inference benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from chapter4_detection.models.nbr_de import NBRDE, register_nbr_de


def module_capacity(model: YOLO) -> dict:
    modules = [m for m in model.model.modules() if isinstance(m, NBRDE)]
    if not modules:
        return {"nbr_total_parameters": 0, "nbr_active_parameters": 0}
    m = modules[0]
    groups = {"query": m.query, "key": m.key, "value": m.value, "local": m.local,
              "difference": m.difference, "gate": m.gate}
    active = {"difference", "gate"}
    active.add("local" if m.mode == "no_reference" else "query")
    if m.mode != "no_reference": active |= {"key", "value"}
    if m.mode == "no_gate": active.discard("gate")
    count = lambda module: sum(p.numel() for p in module.parameters())
    return {"nbr_mode": m.mode, "nbr_total_parameters": count(m) + m.gamma.numel(),
            "nbr_active_parameters": sum(count(groups[k]) for k in active) + m.gamma.numel(),
            "inactive_parameter_groups": sorted(set(groups) - active)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=Path, nargs="+", required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    a = p.parse_args(); register_nbr_de()
    if len(a.models) != len(a.names): raise ValueError("--models and --names lengths differ")
    image = cv2.imread(str(a.image))
    if image is None: raise RuntimeError(f"Unreadable image: {a.image}")
    results = {}
    for name, path in zip(a.names, a.models):
        model = YOLO(str(path))
        device = torch.device(f"cuda:{a.device}")
        timing = {}
        for precision, half in (("fp32", False), ("amp_fp16", True)):
            model.predict(image, imgsz=a.imgsz, device=a.device, half=half, verbose=False)
            for _ in range(a.warmup):
                model.predict(image, imgsz=a.imgsz, device=a.device, half=half, verbose=False)
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
            times=[]
            for _ in range(a.repeats):
                torch.cuda.synchronize(device); start=time.perf_counter()
                model.predict(image, imgsz=a.imgsz, device=a.device, half=half, verbose=False)
                torch.cuda.synchronize(device); times.append((time.perf_counter()-start)*1000)
            timing[precision] = {"end_to_end_ms_mean": float(np.mean(times)),
                                 "end_to_end_ms_sd": float(np.std(times, ddof=1)),
                                 "throughput_images_s": 1000/float(np.mean(times)),
                                 "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))}
        results[name] = {"weights": str(path), "parameters": int(sum(p.numel() for p in model.model.parameters())),
                         "gflops_ultralytics": float(get_flops(model.model, imgsz=a.imgsz)),
                         "timing": timing, **module_capacity(model)}
        del model; torch.cuda.empty_cache()
    output={"protocol": {"gpu": torch.cuda.get_device_name(torch.device(f"cuda:{a.device}")), "device": a.device,
             "image": str(a.image), "original_shape": list(image.shape[:2]), "imgsz": a.imgsz,
             "batch": 1, "warmup_per_precision": a.warmup, "repeats_per_precision": a.repeats, "cuda_synchronize": True,
             "timing_scope": "Ultralytics predict end-to-end; image already decoded; fp32 uses half=False; amp_fp16 uses half=True"}, "models": results}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
