import unittest

import numpy as np

from chapter4_detection.scripts.slicing import (
    Tile,
    class_aware_nms,
    clip_boxes_to_tile,
    make_tiles,
    restore_boxes,
    xyxy_to_yolo,
    yolo_to_xyxy,
)


class SlicingTests(unittest.TestCase):
    def test_expected_4096_tiles(self):
        tiles = make_tiles(4096, 1024, 1024, 128)
        self.assertEqual(len(tiles), 5)
        self.assertEqual((tiles[-1].x1, tiles[-1].x2), (3072, 4096))

    def test_coordinate_roundtrip(self):
        boxes = np.array([[100.25, 10.5, 130.75, 40.5], [900, 900, 1000, 1000]])
        normalized = xyxy_to_yolo(boxes, 1024, 1024)
        np.testing.assert_allclose(yolo_to_xyxy(normalized, 1024, 1024), boxes, atol=1e-9)

    def test_cross_tile_target_is_not_dropped(self):
        box = np.array([[890.0, 100.0, 920.0, 130.0]])
        coverages = []
        for tile in make_tiles(4096, 1024, 1024, 128):
            _, keep, visibility = clip_boxes_to_tile(box, tile, min_visibility=0.5)
            if len(keep):
                coverages.extend(visibility.tolist())
        self.assertTrue(coverages)
        self.assertAlmostEqual(max(coverages), 1.0)

    def test_restore(self):
        tile = Tile(896, 0, 1920, 1024)
        local = np.array([[4, 20, 24, 40]], dtype=float)
        np.testing.assert_array_equal(restore_boxes(local, tile), [[900, 20, 920, 40]])

    def test_class_aware_nms_keeps_different_classes(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 11, 11]])
        keep = class_aware_nms(boxes, np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]), 0.5)
        self.assertEqual(keep.tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
