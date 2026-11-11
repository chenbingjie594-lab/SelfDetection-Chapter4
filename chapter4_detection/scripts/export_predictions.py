#!/usr/bin/env python3
"""Export E0/E1 full-image or E2 sliced predictions in one original-coordinate schema."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from chapter4_detection.scripts.slicing import class_aware_nms, make_tiles, restore_boxes


def synchronize(use_cuda: bool) -> None:
    if use_cuda:
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--mode", choices=("full", "sliced"), required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.conf != 0.001 or args.nms_iou != 0.7:
        raise ValueError("Formal export is frozen at conf=0.001 and nms_iou=0.7")

    model = YOLO(str(args.weights))
    use_cuda = str(args.device).lower() != "cpu" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise RuntimeError(f"No images in {args.images}")
    warm = cv2.imread(str(paths[0]))
    warm = warm if args.mode == "full" else warm[:args.tile_size, :args.tile_size]
    for _ in range(args.warmup):
        model.predict(warm, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou, device=args.device, verbose=False)

    payload, totals, forwards, counts = {}, [], [], []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Unreadable image: {path}")
        height, width = image.shape[:2]
        synchronize(use_cuda)
        total_start = time.perf_counter()
        if args.mode == "full":
            synchronize(use_cuda)
            forward_start = time.perf_counter()
            result = model.predict(image, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
                                   device=args.device, verbose=False)[0]
            synchronize(use_cuda)
            forward_seconds = time.perf_counter() - forward_start
            if result.boxes is None:
                predictions = []
            else:
                predictions = [{"class_id": int(c), "confidence": float(s), "xyxy": b.tolist()}
                               for b, s, c in zip(result.boxes.xyxy.cpu().numpy(),
                                                  result.boxes.conf.cpu().numpy(),
                                                  result.boxes.cls.cpu().numpy().astype(int))]
            tile_count = 1
        else:
            boxes, scores, classes, forward_seconds = [], [], [], 0.0
            tiles = make_tiles(width, height, args.tile_size, args.overlap)
            for tile in tiles:
                crop = image[tile.y1:tile.y2, tile.x1:tile.x2]
                synchronize(use_cuda)
                forward_start = time.perf_counter()
                result = model.predict(crop, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
                                       device=args.device, verbose=False)[0]
                synchronize(use_cuda)
                forward_seconds += time.perf_counter() - forward_start
                if result.boxes is not None and len(result.boxes):
                    boxes.append(restore_boxes(result.boxes.xyxy.cpu().numpy(), tile))
                    scores.append(result.boxes.conf.cpu().numpy())
                    classes.append(result.boxes.cls.cpu().numpy().astype(int))
            if boxes:
                b, s, c = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
                keep = class_aware_nms(b, s, c, args.nms_iou)
                predictions = [{"class_id": int(c[i]), "confidence": float(s[i]), "xyxy": b[i].tolist()} for i in keep]
            else:
                predictions = []
            tile_count = len(tiles)
        synchronize(use_cuda)
        total_seconds = time.perf_counter() - total_start
        totals.append(total_seconds)
        forwards.append(forward_seconds)
        counts.append(len(predictions))
        payload[path.name] = {"width": width, "height": height, "tiles": tile_count,
                              "forward_seconds": forward_seconds, "seconds": total_seconds,
                              "predictions": predictions}

    output = {
        "schema_version": 1, "model": args.model_name, "mode": args.mode,
        "protocol": "all boxes in original 4096x1024 coordinates; conf=0.001; class-aware NMS IoU=0.7",
        "weights": str(args.weights.resolve()), "conf": args.conf, "nms_iou": args.nms_iou,
        "images_count": len(payload), "mean_predictions_per_original": float(np.mean(counts)),
        "mean_forward_seconds_per_original": float(np.mean(forwards)),
        "mean_seconds_per_original": float(np.mean(totals)),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if use_cuda else None,
        "images": payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "images"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
