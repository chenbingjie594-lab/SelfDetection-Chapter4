import unittest
from pathlib import Path

import torch
from ultralytics import YOLO

from chapter4_detection.scripts.model_audit import compatible_transfer


ROOT = Path(__file__).resolve().parents[2]


class ModelTests(unittest.TestCase):
    def test_e1_has_p2_p3_p4_p5_strides_and_loads_weights(self):
        model = YOLO(str(ROOT / "chapter4_detection/models/yolo11s-p2.yaml"))
        self.assertEqual([int(x) for x in model.model.stride], [4, 8, 16, 32])
        report = compatible_transfer(model, ROOT / "yolo11s.pt")
        self.assertGreater(report["matched_parameters"], 0)
        self.assertTrue(report["missing_target_keys"])
        model.model.eval()
        with torch.inference_mode():
            output = model.model(torch.zeros(1, 3, 256, 256))
        self.assertIsNotNone(output)

    def test_e0_forward(self):
        model = YOLO(str(ROOT / "yolo11s.pt"))
        model.model.eval()
        with torch.inference_mode():
            output = model.model(torch.zeros(1, 3, 256, 256))
        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
