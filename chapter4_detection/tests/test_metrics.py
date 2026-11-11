import unittest

import torch
import numpy as np

from chapter4_detection.scripts.evaluate_original import match_predictions
from chapter4_detection.scripts.diagnose_predictions import greedy_matches
from chapter4_detection.scripts.summarize_training import expected_optimizer_updates


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

    def test_diagnostic_matching_is_one_to_one(self):
        matches = greedy_matches(
            np.asarray([0, 0]), np.asarray([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float),
            np.asarray([0]), np.asarray([[0, 0, 10, 10]], dtype=float), np.asarray([0.9]), 0.5, 0.25,
        )
        self.assertEqual(len(matches), 1)

    def test_optimizer_update_schedule_accounts_for_accumulation(self):
        updates = expected_optimizer_updates(samples=138, epochs=76, batch=1, nominal_batch=64, warmup_epochs=3.0)
        self.assertEqual(updates, 184)
        self.assertLess(updates, 138 * 76)


if __name__ == "__main__":
    unittest.main()
