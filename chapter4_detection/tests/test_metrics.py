import unittest

import torch

from chapter4_detection.scripts.evaluate_original import match_predictions


class MetricTests(unittest.TestCase):
    def test_one_to_one_matching(self):
        pred_cls = torch.tensor([0.0, 0.0])
        true_cls = torch.tensor([0.0])
        iou = torch.tensor([[0.9, 0.8]])
        correct = match_predictions(pred_cls, true_cls, iou)
        self.assertTrue(correct[0, 0])
        self.assertFalse(correct[1, 0])

    def test_class_mismatch(self):
        correct = match_predictions(torch.tensor([1.0]), torch.tensor([0.0]), torch.tensor([[0.9]]))
        self.assertFalse(correct.any())


if __name__ == "__main__":
    unittest.main()
