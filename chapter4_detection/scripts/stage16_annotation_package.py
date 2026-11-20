#!/usr/bin/env python3
"""Build a prediction-free expert annotation review package (read-only inputs)."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


CLASSES = {0: "Flash point", 1: "Big black spots"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_yolo(path: Path, width: int, height: int):
    boxes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = map(float, line.split()[:5])
        boxes.append((int(c), ((x-w/2)*width, (y-h/2)*height,
                               (x+w/2)*width, (y+h/2)*height)))
    return boxes


def parse_xml(path: Path):
    if path.suffix.lower() != ".xml" or not path.exists():
        return None, []
    root = ET.parse(path).getroot()
    size = root.find("size")
    wh = (int(size.findtext("width")), int(size.findtext("height"))) if size is not None else None
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        boxes.append((obj.findtext("name", ""), tuple(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return wh, boxes


def expanded(box, width, height, factor=8, min_side=160):
    x1, y1, x2, y2 = box
    cx, cy = (x1+x2)/2, (y1+y2)/2
    side = max(min_side, (x2-x1)*factor, (y2-y1)*factor)
    return (max(0, int(cx-side/2)), max(0, int(cy-side/2)),
            min(width, int(cx+side/2)), min(height, int(cy+side/2)))


def panel(image, crop, box, color, title, size=(700, 430)):
    region = image.crop(crop).convert("RGB")
    sx, sy = size[0] / region.width, size[1] / region.height
    region = region.resize(size)
    d = ImageDraw.Draw(region)
    x1, y1, x2, y2 = box
    d.rectangle(((x1-crop[0])*sx, (y1-crop[1])*sy, (x2-crop[0])*sx, (y2-crop[1])*sy), outline=color, width=4)
    d.rectangle((0, 0, size[0], 30), fill=(255, 255, 255))
    d.text((8, 7), title, fill=(0, 0, 0))
    return region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--review-table", type=Path, required=True)
    ap.add_argument("--xml-trace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plate_dir = args.output / "annotation_review"
    plate_dir.mkdir(exist_ok=True)
    manifest = json.loads((args.dataset / "split_manifest.json").read_text())
    by_name = {r["image_name"]: r for r in manifest["records"]}
    traces = {r["image"]: r for r in csv.DictReader(args.xml_trace.open())}
    reviews = list(csv.DictReader(args.review_table.open()))
    audit, template = [], []
    for idx, row in enumerate(reviews):
        name, gt_id = row["image"], int(row["gt_id"])
        rec, trace = by_name[name], traces[name]
        image_path, label_path = Path(rec["image"]), Path(rec["label"])
        im = Image.open(image_path).convert("RGB")
        boxes = parse_yolo(label_path, *im.size)
        cls, ybox = boxes[gt_id]
        xml_path = Path(trace["xml"]) if trace.get("xml") else None
        xml_wh, xml_boxes = parse_xml(xml_path) if xml_path else (None, [])
        # Pair only for neutral presentation; no correctness decision is made.
        xbox = None
        if xml_boxes:
            def center_dist(item):
                b=item[1]; return abs((b[0]+b[2])-(ybox[0]+ybox[2])) + abs((b[1]+b[3])-(ybox[1]+ybox[3]))
            xbox = min(xml_boxes, key=center_dist)[1]
        crop = expanded(ybox, *im.size)
        header = Image.new("RGB", (1400, 100), "white")
        hd = ImageDraw.Draw(header)
        hd.text((10, 10), f"{name} | GT {gt_id} | {CLASSES[cls]}", fill="black")
        hd.text((10, 38), f"Current YOLO source: {row['provenance']} | xyxy: {[round(v,2) for v in ybox]}", fill="black")
        hd.text((10, 66), "Blind review: no predictions, AP, or agreement score are shown.", fill="black")
        yp = panel(im, crop, ybox, (230, 30, 30), "View A — current YOLO annotation")
        if xbox is not None:
            xp = panel(im, crop, xbox, (30, 80, 230), "View B — historical XML annotation")
        else:
            xp = Image.new("RGB", yp.size, "white")
            ImageDraw.Draw(xp).text((20, 30), "View B — no original XML available\nHistorical evidence is audited separately.", fill="black")
        canvas = Image.new("RGB", (1400, 530), "white")
        canvas.paste(header, (0, 0)); canvas.paste(yp, (0, 100)); canvas.paste(xp, (700, 100))
        plate_name = f"{idx:02d}_{Path(name).stem}_gt{gt_id}.jpg"
        canvas.save(plate_dir / plate_name, quality=94)
        x1,y1,x2,y2=ybox
        near = x1 <= 2 or y1 <= 2 or x2 >= im.width-2 or y2 >= im.height-2
        audit.append({
            "image": name, "gt_id": gt_id, "split": rec["split"], "class": CLASSES[cls],
            "image_exists": image_path.exists(), "label_exists": label_path.exists(),
            "image_wh": f"{im.width}x{im.height}", "manifest_wh": f"{rec['width']}x{rec['height']}",
            "dimensions_match": im.size == (rec["width"], rec["height"]),
            "bbox_in_bounds": x1 >= 0 and y1 >= 0 and x2 <= im.width and y2 <= im.height,
            "near_image_boundary": near, "source_group": rec["source_group"],
            "same_source_group_members": sum(r["source_group"] == rec["source_group"] for r in manifest["records"]),
            "xml_or_evidence": str(xml_path or ""), "xml_exists": bool(xml_path and xml_path.exists()),
            "xml_is_original_annotation": bool(xml_path and xml_path.suffix.lower() == ".xml"),
            "xml_dimensions_match": xml_wh == im.size if xml_wh else "UNKNOWN",
            "plate": plate_name, "image_sha256": sha256(image_path), "label_sha256": sha256(label_path),
            "audit_conclusion": "PENDING_EXPERT_REVIEW",
        })
        template.append({"image": name, "gt_id": gt_id, "reviewer_id": "", "review_status": "PENDING_EXPERT_REVIEW",
                         "object_valid": "", "class_correct": "", "boundary_quality": "", "corrected_bbox": "",
                         "confidence": "", "reason": "", "notes": ""})
    def write_csv(path, rows):
        with path.open("w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_csv(args.output / "annotation_review_audit.csv", audit)
    write_csv(args.output / "EXPERT_REVIEW_TEMPLATE.csv", template)
    meta={"inputs": {str(p): sha256(p) for p in [args.review_table,args.xml_trace,args.dataset/"split_manifest.json"]},
          "review_items": len(reviews), "plates": len(list(plate_dir.glob("*.jpg"))),
          "expert_conclusions_completed": 0, "status": "PENDING_EXPERT_REVIEW"}
    (args.output / "annotation_package_metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
