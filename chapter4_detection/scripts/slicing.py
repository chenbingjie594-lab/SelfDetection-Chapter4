"""Native-scale tiling and coordinate transforms used by E2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tile:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_tiles(width: int, height: int, tile_size: int = 1024, overlap: int = 128) -> list[Tile]:
    return [
        Tile(x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in axis_starts(height, tile_size, overlap)
        for x in axis_starts(width, tile_size, overlap)
    ]


def yolo_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
    out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
    out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
    out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
    return out


def xyxy_to_yolo(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    out = np.empty_like(boxes)
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / width
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / height
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
    return out


def clip_boxes_to_tile(boxes_xyxy: np.ndarray, tile: Tile, min_visibility: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip original-image boxes and return tile boxes, kept indices and visible fractions."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], tile.x1, tile.x2)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], tile.y1, tile.y2)
    original_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    clipped_area = np.maximum(0, clipped[:, 2] - clipped[:, 0]) * np.maximum(0, clipped[:, 3] - clipped[:, 1])
    visibility = np.divide(clipped_area, original_area, out=np.zeros_like(clipped_area), where=original_area > 0)
    keep = np.flatnonzero((clipped_area > 0) & (visibility >= min_visibility))
    local = clipped[keep]
    local[:, [0, 2]] -= tile.x1
    local[:, [1, 3]] -= tile.y1
    return local, keep, visibility[keep]


def restore_boxes(boxes_xyxy: np.ndarray, tile: Tile) -> np.ndarray:
    out = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4).copy()
    out[:, [0, 2]] += tile.x1
    out[:, [1, 3]] += tile.y1
    return out


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    one = np.asarray(one, dtype=np.float64)
    many = np.asarray(many, dtype=np.float64).reshape(-1, 4)
    inter_x1 = np.maximum(one[0], many[:, 0])
    inter_y1 = np.maximum(one[1], many[:, 1])
    inter_x2 = np.minimum(one[2], many[:, 2])
    inter_y2 = np.minimum(one[3], many[:, 3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area1 = max(0, one[2] - one[0]) * max(0, one[3] - one[1])
    area2 = np.maximum(0, many[:, 2] - many[:, 0]) * np.maximum(0, many[:, 3] - many[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-12)


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_threshold: float) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    classes = np.asarray(classes, dtype=np.int64)
    kept: list[int] = []
    for cls in np.unique(classes):
        order = np.flatnonzero(classes == cls)
        order = order[np.argsort(scores[order])[::-1]]
        while order.size:
            current = int(order[0])
            kept.append(current)
            order = order[1:]
            if order.size:
                order = order[box_iou(boxes[current], boxes[order]) <= iou_threshold]
    return np.asarray(sorted(kept, key=lambda i: scores[i], reverse=True), dtype=np.int64)
