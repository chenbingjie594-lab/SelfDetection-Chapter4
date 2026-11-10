#!/usr/bin/env python3
"""Run deterministic E2 tiling, restore boxes, and merge per original image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from chapter4_detection.scripts.slicing import class_aware_nms, make_tiles, restore_boxes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    payload, elapsed = {}, []
    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        boxes, scores, classes = [], [], []
        start = time.perf_counter()
        tiles = make_tiles(width, height, args.tile_size, args.overlap)
        for tile in tiles:
            crop = image[tile.y1:tile.y2, tile.x1:tile.x2]
            result = model.predict(crop, imgsz=args.tile_size, conf=args.conf, iou=args.nms_iou, device=args.device, verbose=False)[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes.append(restore_boxes(result.boxes.xyxy.cpu().numpy(), tile))
            scores.append(result.boxes.conf.cpu().numpy())
            classes.append(result.boxes.cls.cpu().numpy().astype(int))
        if boxes:
            box_array, score_array, class_array = np.concatenate(boxes), np.concatenate(scores), np.concatenate(classes)
            keep = class_aware_nms(box_array, score_array, class_array, args.nms_iou)
            predictions = [
                {"class_id": int(class_array[i]), "confidence": float(score_array[i]), "xyxy": box_array[i].tolist()}
                for i in keep
            ]
        else:
            predictions = []
        duration = time.perf_counter() - start
        elapsed.append(duration)
        payload[image_path.name] = {"width": width, "height": height, "tiles": len(tiles), "seconds": duration, "predictions": predictions}
    output = {"protocol": "original->fixed tiles->restore->whole-image class-aware NMS", "mean_seconds_per_original": float(np.mean(elapsed)), "images": payload}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(payload), "mean_seconds_per_original": output["mean_seconds_per_original"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
