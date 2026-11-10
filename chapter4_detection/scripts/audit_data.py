#!/usr/bin/env python3
"""Audit the frozen Real-only split before any chapter-4 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SYNTHETIC_MARKERS = ("synthetic", "generated", "dwbg_", "bootstrap", "random_matched", "poisson")
EXPECTED = {"train_images": 138, "val_images": 46, "train_instances": 168, "train_classes": {0: 88, 1: 80}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_image_id(name: str) -> str:
    """Return the physical-source key encoded by known Plastic_Bomo names."""
    stem = Path(name).stem.lower()
    stem = re.sub(r"_pre_part_[01]$", "", stem)
    return stem


def load_data_yaml(path: Path) -> tuple[Path, dict[int, str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(data["path"])
    names = data.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    if names != {0: "Flash point", 1: "Big black spots"}:
        raise ValueError(f"Unexpected class map: {names}")
    return root, names


def image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def parse_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO-HBB fields, got {len(fields)}")
        cls = int(fields[0])
        coords = tuple(float(x) for x in fields[1:])
        if cls not in (0, 1) or any(not 0.0 <= x <= 1.0 for x in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{path}:{line_no}: invalid class or normalized box")
        rows.append((cls, *coords))
    return rows


def audit_split(root: Path, split: str) -> tuple[list[dict], Counter]:
    images = image_files(root / "images" / split)
    records, classes = [], Counter()
    for image_path in images:
        label_path = root / "labels" / split / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}")
        rows = parse_labels(label_path)
        classes.update(row[0] for row in rows)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unreadable image: {image_path}")
        height, width = image.shape[:2]
        cid = canonical_image_id(image_path.name)
        records.append(
            {
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "image_name": image_path.name,
                "image_sha256": sha256(image_path),
                "label_sha256": sha256(label_path),
                "canonical_image_id": cid,
                "source_group": cid,
                "width": width,
                "height": height,
                "instances": len(rows),
                "class_counts": dict(Counter(row[0] for row in rows)),
            }
        )
    extra_labels = sorted(
        p.name for p in (root / "labels" / split).glob("*.txt") if not (root / "images" / split / f"{p.stem}.jpg").exists()
        and not any((root / "images" / split / f"{p.stem}{s}").exists() for s in IMAGE_SUFFIXES)
    )
    if extra_labels:
        raise ValueError(f"Labels without images in {split}: {extra_labels[:10]}")
    return records, classes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-audit", type=Path)
    args = parser.parse_args()

    root, names = load_data_yaml(args.data.resolve())
    train, train_classes = audit_split(root, "train")
    val, val_classes = audit_split(root, "val")
    all_records = train + val
    failures, warnings = [], []

    checks = {
        "train_images_138": len(train) == EXPECTED["train_images"],
        "val_images_46": len(val) == EXPECTED["val_images"],
        "train_instances_168": sum(x["instances"] for x in train) == EXPECTED["train_instances"],
        "train_class_counts_88_80": dict(train_classes) == EXPECTED["train_classes"],
        "all_images_4096x1024": all((x["width"], x["height"]) == (4096, 1024) for x in all_records),
        "no_byte_identical_train_val": not ({x["image_sha256"] for x in train} & {x["image_sha256"] for x in val}),
        "no_synthetic_name_markers": not any(
            marker in Path(x["image"]).name.lower() for x in all_records for marker in SYNTHETIC_MARKERS
        ),
    }
    for key, ok in checks.items():
        if not ok:
            failures.append(key)

    train_groups = defaultdict(list)
    val_groups = defaultdict(list)
    for record in train:
        train_groups[record["source_group"]].append(record["image_name"])
    for record in val:
        val_groups[record["source_group"]].append(record["image_name"])
    overlaps = [
        {"source_group": group, "train": train_groups[group], "val": val_groups[group]}
        for group in sorted(set(train_groups) & set(val_groups))
    ]
    checks["no_known_source_group_train_val_overlap"] = not overlaps
    if overlaps:
        failures.append("no_known_source_group_train_val_overlap")

    historical = None
    if args.historical_audit:
        historical = json.loads(args.historical_audit.read_text(encoding="utf-8"))
        claimed = historical.get("real_source", {})
        if not claimed:
            warnings.append("historical audit has no real_source split hashes")

    manifest_digest = hashlib.sha256(
        "\n".join(f"{x['split']}\t{x['image_name']}\t{x['image_sha256']}\t{x['label_sha256']}" for x in all_records).encode()
    ).hexdigest()
    report = {
        "status": "PASS" if not failures else "BLOCKED",
        "reason": "all mandatory gates passed" if not failures else "mandatory data gate failure; formal training forbidden",
        "data_yaml": str(args.data.resolve()),
        "data_root": str(root),
        "class_names": names,
        "summary": {
            "train_images": len(train),
            "val_images": len(val),
            "train_instances": sum(x["instances"] for x in train),
            "val_instances": sum(x["instances"] for x in val),
            "train_class_counts": dict(train_classes),
            "val_class_counts": dict(val_classes),
            "sizes": dict(Counter(f"{x['width']}x{x['height']}" for x in all_records)),
        },
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "known_source_group_overlaps": overlaps,
        "manifest_sha256": manifest_digest,
        "historical_split_hash_claims": historical.get("real_source") if historical else None,
        "records": all_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "summary", "failures", "known_source_group_overlaps")}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
