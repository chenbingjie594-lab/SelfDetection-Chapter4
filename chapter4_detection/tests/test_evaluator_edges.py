import numpy as np
import torch

from chapter4_detection.scripts.evaluate_original import match_predictions
from ultralytics.utils.metrics import box_iou


def test_empty_predictions_and_empty_targets():
    assert match_predictions(torch.empty(0), torch.tensor([0.0]), torch.empty((1, 0))).shape == (0, 10)
    assert match_predictions(torch.tensor([0.0]), torch.empty(0), torch.empty((0, 1))).sum() == 0


def test_duplicate_predictions_only_one_matches():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    correct = match_predictions(torch.tensor([0.0, 0.0]), torch.tensor([0.0]), box_iou(gt, pred))
    assert np.all(correct.sum(0).numpy() == 1)


def test_wrong_class_does_not_match():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    correct = match_predictions(torch.tensor([1.0]), torch.tensor([0.0]), box_iou(boxes, boxes))
    assert not correct.any()
