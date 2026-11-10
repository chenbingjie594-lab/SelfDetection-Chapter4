#!/usr/bin/env python3
"""Create E2 train tiles from the frozen real-only train split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from chapter4_detection.scripts.slicing import clip_boxes_to_tile, make_tiles, xyxy_to_yolo, yolo_to_xyxy


def require_pass(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise RuntimeError(f"Data audit is {report.get('status')}; refusing to generate formal E2 data")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    args = parser.parse_args()
    report = require_pass(args.audit_report)
    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(data["path"])
    expected_train = {Path(x["image"]).resolve() for x in report["records"] if x["split"] == "train"}

    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    image_out, label_out = args.output / "images/train", args.output / "labels/train"
    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    manifest, coverage = [], Counter()
    for image_path in sorted((root / "images/train").glob("*")):
        if not image_path.is_file():
            continue
        if image_path.resolve() not in expected_train:
            raise RuntimeError(f"Unapproved train image: {image_path}")
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        rows = [line.split() for line in (root / "labels/train" / f"{image_path.stem}.txt").read_text().splitlines() if line.strip()]
        classes = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
        boxes = yolo_to_xyxy(np.asarray([[float(x) for x in row[1:]] for row in rows]), width, height)
        per_target_max = np.zeros(len(boxes), dtype=float)
        for tile_index, tile in enumerate(make_tiles(width, height, args.tile_size, args.overlap)):
            local, keep, visibility = clip_boxes_to_tile(boxes, tile, args.min_visibility)
            per_target_max[keep] = np.maximum(per_target_max[keep], visibility)
            name = f"{image_path.stem}__x{tile.x1}_y{tile.y1}"
            cv2.imwrite(str(image_out / f"{name}.jpg"), image[tile.y1:tile.y2, tile.x1:tile.x2])
            normalized = xyxy_to_yolo(local, tile.width, tile.height)
            with (label_out / f"{name}.txt").open("w", encoding="utf-8") as handle:
                for cls, box in zip(classes[keep], normalized):
                    handle.write(f"{cls} " + " ".join(f"{v:.8f}" for v in box) + "\n")
            manifest.append({"tile": name, "source": str(image_path), "xyxy": [tile.x1, tile.y1, tile.x2, tile.y2], "instances": len(keep)})
        if np.any(per_target_max < args.min_visibility):
            raise RuntimeError(f"Target coverage gate failed for {image_path}: {per_target_max.tolist()}")
        coverage.update({"targets": len(boxes), "fully_covered": int(np.sum(per_target_max >= 1.0 - 1e-9))})

    # Reuse the untouched full-image validation split via symlinks; no val slicing is used for official metrics.
    (args.output / "images").mkdir(exist_ok=True)
    (args.output / "labels").mkdir(exist_ok=True)
    (args.output / "images/val").symlink_to((root / "images/val").resolve(), target_is_directory=True)
    (args.output / "labels/val").symlink_to((root / "labels/val").resolve(), target_is_directory=True)
    dataset_yaml = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val", "names": data["names"]}
    (args.output / "data.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    summary = {"tiles": len(manifest), "coverage": dict(coverage), "tile_size": args.tile_size, "overlap": args.overlap, "manifest": manifest}
    (args.output / "slice_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tiles", "coverage", "tile_size", "overlap")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
