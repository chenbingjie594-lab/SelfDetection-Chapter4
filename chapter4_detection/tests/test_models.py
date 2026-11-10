import unittest
import tempfile
from pathlib import Path

import torch
from ultralytics import YOLO

from chapter4_detection.scripts.model_audit import semantic_transfer


ROOT = Path(__file__).resolve().parents[2]


class ModelTests(unittest.TestCase):
    def test_e1_has_p2_p3_p4_p5_strides_and_loads_weights(self):
        model = YOLO(str(ROOT / "chapter4_detection/models/yolo11s-p2.yaml"))
        self.assertEqual([int(x) for x in model.model.stride], [4, 8, 16, 32])
        report = semantic_transfer(model, ROOT / "yolo11s.pt")
        self.assertGreater(report["matched_parameters"], 0)
        self.assertTrue(report["value_verification_passed"])
        mapping = {row["source"]: row["target"] for row in report["explicit_mapping"]}
        self.assertEqual(mapping["model.17.conv.weight"], "model.23.conv.weight")
        self.assertEqual(mapping["model.19.cv1.conv.weight"], "model.25.cv1.conv.weight")
        self.assertEqual(mapping["model.20.conv.weight"], "model.26.conv.weight")
        self.assertEqual(mapping["model.22.cv1.conv.weight"], "model.28.cv1.conv.weight")
        self.assertEqual(mapping["model.23.cv2.0.0.conv.weight"], "model.29.cv2.1.0.conv.weight")
        self.assertNotIn("model.29.cv2.0.0.conv.weight", set(mapping.values()))
        self.assertTrue(report["missing_target_keys"])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "initialized.pt"
            expected = model.model.state_dict()["model.23.conv.weight"].clone()
            model.save(checkpoint)
            reloaded = YOLO(str(checkpoint))
            self.assertTrue(torch.equal(expected, reloaded.model.state_dict()["model.23.conv.weight"]))
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
