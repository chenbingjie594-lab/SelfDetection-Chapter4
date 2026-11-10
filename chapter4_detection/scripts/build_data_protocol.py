#!/usr/bin/env python3
"""Freeze a group-isolated, real-only Chapter 4 Plastic_Bomo protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_sha(data: object) -> str:
    return sha256_bytes(json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def image_fingerprint(path: Path) -> tuple[str, int, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Unreadable image: {path}")
    height, width = image.shape[:2]
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 8), interpolation=cv2.INTER_AREA)
    comparisons = gray[:, :-1] > gray[:, 1:]
    dhash = sum(int(comparisons[j, i]) << (i * 8 + j) for i in range(31) for j in range(8))
    return sha256_bytes(image.tobytes()), width, height, dhash


def labels(path: Path) -> tuple[list[int], list[str]]:
    classes, errors = [], []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            cls = int(fields[0])
            values = [float(v) for v in fields[1:]]
        except (ValueError, IndexError):
            errors.append(f"line {index}: parse error")
            continue
        if len(fields) != 5 or cls not in (0, 1) or not all(np.isfinite(values)):
            errors.append(f"line {index}: invalid YOLO HBB row")
        elif not all(0 <= v <= 1 for v in values) or values[2] <= 0 or values[3] <= 0:
            errors.append(f"line {index}: coordinates outside valid range")
        classes.append(cls)
    return classes, errors


def xml_origin(xml_root: Path, stem: str) -> tuple[str, dict] | None:
    xml_path = xml_root / f"{stem}.xml"
    if not xml_path.exists():
        return None
    root = ET.parse(xml_path).getroot()
    filename = (root.findtext("filename") or "").strip()
    original_path = (root.findtext("path") or "").strip()
    if not filename or not original_path:
        return None
    identity = f"xml:{original_path}|{filename}"
    return "src_" + sha256_bytes(identity.encode())[:16], {
        "method": "CONFIRMED_ORIGINAL_XML",
        "evidence": str(xml_path),
        "original_filename": filename,
        "original_capture_path": original_path,
    }


def black_evidence(evidence_path: Path) -> dict[str, dict]:
    if not evidence_path.exists():
        return {}
    report = json.loads(evidence_path.read_text(encoding="utf-8"))
    result = {}
    for row in report.get("records", []):
        candidates = [c for c in row.get("candidates", []) if c.get("exact_file_match")]
        if candidates:
            result[Path(row["frozen_image"]).stem] = candidates[0]
    return result


def choose_validation(groups: dict[str, list[dict]], seed: int, total_counts: Counter, target_images: int) -> set[str]:
    """Deterministic seeded search; source groups are indivisible and labels only stratify the split."""
    group_ids, rng = sorted(groups), random.Random(seed)
    target_classes = (total_counts[0] * 0.25, total_counts[1] * 0.25)
    best: tuple[tuple, set[str]] | None = None
    for _ in range(25000):
        ordered = rng.sample(group_ids, len(group_ids))
        chosen, images = set(), 0
        for group in ordered:
            size = len(groups[group])
            if images + size <= target_images:
                chosen.add(group)
                images += size
            if images == target_images:
                break
        counts = [sum(r["class_counts"][str(cls)] for g in chosen for r in groups[g]) for cls in (0, 1)]
        score = (abs(images - target_images), abs(counts[0] - target_classes[0]) + abs(counts[1] - target_classes[1]),
                 abs(counts[0] - target_classes[0]), tuple(sorted(chosen)))
        if best is None or score < best[0]:
            best = score, chosen
    assert best is not None
    return best[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--xml-root", type=Path, required=True)
    parser.add_argument("--black-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol: {args.output}")

    spec = yaml.safe_load(args.source_yaml.read_text(encoding="utf-8"))
    root = Path(spec["path"])
    rebuilt_black = black_evidence(args.black_evidence)
    records, failures = [], []
    for old_split in ("train", "val"):
        for image_path in sorted((root / spec[old_split]).glob("*")):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = root / f"labels/{old_split}/{image_path.stem}.txt"
            if not label_path.exists():
                failures.append(f"missing label: {label_path}")
                continue
            class_ids, label_errors = labels(label_path)
            failures.extend(f"{image_path.name}: {e}" for e in label_errors)
            decoded_sha, width, height, dhash = image_fingerprint(image_path)
            origin = xml_origin(args.xml_root, image_path.stem)
            if origin:
                source_group, provenance = origin
                canonical_id = provenance["original_filename"].rsplit(".", 1)[0]
            elif image_path.stem.startswith("blackinst_"):
                source_number = image_path.stem.removeprefix("blackinst_")
                source_group = f"black_source_{source_number}"
                canonical_id = source_group
                if image_path.stem in rebuilt_black:
                    candidate = rebuilt_black[image_path.stem]
                    provenance = {"method": "CONFIRMED_EXACT_HISTORICAL_RECONSTRUCTION", "evidence": str(args.black_evidence),
                                  "original_source": candidate["source"], "original_source_sha256": candidate["source_sha256"]}
                else:
                    provenance = {"method": "SUPPORTED_DETERMINISTIC_HISTORICAL_EXPORT_ID",
                                  "evidence": "blackinst_<N> is emitted from the unique source basename <N>; exact source bytes are no longer present",
                                  "original_source": f"Black instance/{source_number}.<jpg-or-png>"}
            else:
                source_group, canonical_id = f"unknown_{image_path.stem}", image_path.stem
                provenance = {"method": "UNKNOWN"}
            counts = Counter(class_ids)
            records.append({
                "image_name": image_path.name, "canonical_image_id": canonical_id, "source_group": source_group,
                "provenance": provenance, "old_split": old_split, "original_path": str(image_path),
                "original_label_path": str(label_path), "image_sha256": sha256_file(image_path),
                "decoded_rgb_sha256": decoded_sha, "label_sha256": sha256_file(label_path),
                "width": width, "height": height, "dhash": str(dhash),
                "instances": len(class_ids), "class_counts": {"0": counts[0], "1": counts[1]},
            })

    if len(records) != 184:
        failures.append(f"expected 184 images, found {len(records)}")
    unknown = [r["image_name"] for r in records if r["provenance"]["method"] == "UNKNOWN"]
    if unknown:
        failures.append(f"unknown provenance: {unknown}")
    for r in records:
        if (r["width"], r["height"]) != (4096, 1024):
            failures.append(f"unexpected dimensions: {r['image_name']} {r['width']}x{r['height']}")

    groups = defaultdict(list)
    for row in records:
        groups[row["source_group"]].append(row)
    totals = Counter()
    for row in records:
        totals.update({0: row["class_counts"]["0"], 1: row["class_counts"]["1"]})
    validation_groups = choose_validation(groups, args.seed, totals, round(len(records) * 0.25))
    for row in records:
        row["split"] = "val" if row["source_group"] in validation_groups else "train"

    # Exact byte/decoded duplicates must never cross splits. dHash is diagnostic only.
    for field in ("image_sha256", "decoded_rgb_sha256"):
        values = defaultdict(list)
        for row in records:
            values[row[field]].append(row)
        for digest, same in values.items():
            if len({r["split"] for r in same}) > 1:
                failures.append(f"cross-split {field} duplicate {digest}: {[r['image_name'] for r in same]}")
    cross_source = [g for g, rows in groups.items() if len({r["split"] for r in rows}) > 1]
    if cross_source:
        failures.append(f"cross-split source groups: {cross_source}")
    near_pairs = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            distance = (int(left["dhash"]) ^ int(right["dhash"])).bit_count()
            if distance <= 8:
                near_pairs.append({"left": left["image_name"], "right": right["image_name"], "distance": distance,
                                   "cross_split": left["split"] != right["split"],
                                   "same_source_group": left["source_group"] == right["source_group"],
                                   "disposition": "DIAGNOSTIC_ONLY_NOT_PROVEN_LINEAGE"})

    for row in records:
        split = row["split"]
        image_dst = args.output / "images" / split / row["image_name"]
        label_dst = args.output / "labels" / split / f"{Path(row['image_name']).stem}.txt"
        image_dst.parent.mkdir(parents=True, exist_ok=True)
        label_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["original_path"], image_dst)
        shutil.copy2(row["original_label_path"], label_dst)
        row["image"] = str(image_dst.resolve())
        row["label"] = str(label_dst.resolve())

    split_counts = {}
    for split in ("train", "val"):
        subset = [r for r in records if r["split"] == split]
        split_counts[split] = {"images": len(subset), "instances": sum(r["instances"] for r in subset),
                               "class_0": sum(r["class_counts"]["0"] for r in subset),
                               "class_1": sum(r["class_counts"]["1"] for r in subset),
                               "source_groups": len({r["source_group"] for r in subset})}
    manifest_core = {"protocol_version": "chapter4_real_only_v1", "seed": args.seed,
                     "source_yaml": str(args.source_yaml.resolve()), "records": records}
    protocol_hash = canonical_json_sha(manifest_core)
    manifest = {**manifest_core, "protocol_sha256": protocol_hash, "statistics": split_counts}
    manifest_path = args.output / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_yaml = {"path": str(args.output.resolve()), "train": "images/train", "val": "images/val", "names": spec["names"]}
    (args.output / "data.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    audit = {"status": "PASS" if not failures else "BLOCKED", "failures": failures,
             "protocol_sha256": protocol_hash, "manifest_sha256": sha256_file(manifest_path),
             "summary": split_counts, "source_groups": len(groups), "unknown_provenance": unknown,
             "cross_split_source_groups": cross_source, "near_duplicate_diagnostics": near_pairs,
             "synthetic_images": 0, "records": records}
    (args.output / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence_counts = Counter(r["provenance"]["method"] for r in records)
    protocol_md = f"""# Chapter 4 Real-only Data Protocol v1

- Status: **{audit['status']}**
- Protocol SHA256: `{protocol_hash}`
- Split seed: `{args.seed}`
- Source images: 184 real images; synthetic images: 0
- Train: {split_counts['train']}
- Validation: {split_counts['val']}
- Source groups: {len(groups)}; cross-split groups: {len(cross_source)}
- Provenance evidence: {dict(evidence_counts)}
- Near-duplicate candidates (dHash <= 8): {len(near_pairs)}. These are diagnostics only and are not treated as lineage without provenance evidence.

The split was produced before any model training. Source groups are indivisible. Original XML
`filename/path` fields define the 149 flash-image source groups. Black-instance groups use the
historical deterministic exporter identity; 25/35 are additionally verified by exact historical
reconstruction. The remaining ten retain a lower evidence grade in the manifest rather than being
misreported as byte-level reconstructions.
"""
    (args.output / "DATA_PROTOCOL.md").write_text(protocol_md, encoding="utf-8")
    print(json.dumps({"status": audit["status"], "statistics": split_counts, "protocol_sha256": protocol_hash,
                      "evidence": dict(evidence_counts), "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
