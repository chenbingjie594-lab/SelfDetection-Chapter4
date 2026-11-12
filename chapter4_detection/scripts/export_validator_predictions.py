#!/usr/bin/env python3
"""Export predictions from the exact Ultralytics validation data path."""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.models.yolo.detect.val import DetectionValidator

class PredictionCaptureValidator(DetectionValidator):
    """Capture native-coordinate tensors before any lossy text serialization."""

    captured: dict[str, list[dict]] = {}

    def update_metrics(self, preds, batch) -> None:
        offset = self.seen
        for index, pred in enumerate(preds):
            pbatch = self._prepare_batch(index, batch)
            predn = self._prepare_pred(pred)
            # Preserve the validator's IoU geometry exactly. ops.scale_boxes()
            # clips to the original border; validation matches before clipping,
            # and clipping can flip tiny border boxes across an IoU threshold.
            boxes = predn["bboxes"].clone()
            (gain_y, gain_x), (pad_x, pad_y) = pbatch["ratio_pad"]
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / gain_x
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / gain_y
            self.captured[Path(pbatch["im_file"]).name] = [
                {"class_id": int(c), "confidence": float(s), "xyxy": b.tolist()}
                for b, s, c in zip(boxes.cpu(), predn["conf"].cpu(), predn["cls"].cpu())
            ]
        assert self.seen == offset
        super().update_metrics(preds, batch)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--device", default="0")
    p.add_argument("--name", required=True)
    a = p.parse_args()
    PredictionCaptureValidator.captured = {}
    metrics = YOLO(str(a.weights)).val(
        validator=PredictionCaptureValidator,
        data=str(a.data), imgsz=a.imgsz, batch=1, device=a.device,
        conf=0.001, iou=0.7, max_det=300, plots=False,
    )
    records = {}
    for image_path in sorted(x for x in a.images.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Unreadable image: {image_path}")
        h, w = image.shape[:2]
        predictions = PredictionCaptureValidator.captured.get(image_path.name, [])
        records[image_path.name] = {"width": w, "height": h, "predictions": predictions}
    payload = {
        "schema_version": 1, "model": a.name, "mode": "validator_full",
        "protocol": "Ultralytics validator; conf=0.001; NMS IoU=0.7; max_det=300",
        "weights": str(a.weights.resolve()), "images_count": len(records),
        "validator_metrics": {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)},
        "images": records,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
