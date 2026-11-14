from pathlib import Path

import pytest
import torch
from ultralytics import YOLO

from chapter4_detection.models.nbr_de_r import NBRDER, register_nbr_de_r


def test_forward_shape_identity_and_backward():
    module = NBRDER(16).train()
    x = torch.randn(2, 16, 13, 19, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)
    y.square().mean().backward()
    assert module.gamma.grad is not None and torch.isfinite(module.gamma.grad).all()


def test_reliability_bounds_and_monotonicity():
    module = NBRDER(16)
    u = torch.linspace(0, 10, 101)
    r = module.reliability(u)
    assert torch.all((r >= 0) & (r <= 1))
    assert torch.all(r[1:] <= r[:-1])


def test_background_blending_formula():
    module = NBRDER(16).eval()
    x = torch.randn(1, 16, 9, 13)
    with torch.no_grad():
        attention, local, _, trust, blended, _ = module.references(x)
    assert torch.allclose(blended, trust * attention + (1 - trust) * local)


def test_boundary_sampling_has_no_wraparound():
    x = torch.zeros(1, 1, 7, 7)
    x[..., 0, 0] = 1
    sampled = NBRDER._sample(x, -2, 0, 6)
    assert sampled[..., -1, -1].item() == 0
    assert sampled[..., 0, 0].item() == 0


def test_feature_export_is_finite():
    module = NBRDER(16).eval(); module.export_features = True
    with torch.no_grad(): module(torch.randn(1, 16, 11, 17))
    assert set(module.last_features) == {"F", "B_att", "B_local", "U", "r", "B", "D", "G", "contribution", "attention_weights"}
    assert all(torch.isfinite(value).all() for value in module.last_features.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_numerical_stability():
    module = NBRDER(32).cuda().train()
    x = torch.randn(1, 32, 19, 31, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16): y = module(x)
    assert torch.isfinite(y).all(); y.mean().backward()


def test_yaml_build_and_checkpoint_reload(tmp_path: Path):
    register_nbr_de_r()
    model = YOLO("chapter4_detection/models/yolo11s_nbr_de_r.yaml")
    modules = [m for m in model.model.modules() if isinstance(m, NBRDER)]
    assert len(modules) == 1 and modules[0].gamma.item() == 0
    checkpoint = tmp_path / "e4.pt"; model.save(checkpoint)
    reloaded = YOLO(str(checkpoint))
    loaded = [m for m in reloaded.model.modules() if isinstance(m, NBRDER)]
    assert len(loaded) == 1
    assert loaded[0].offsets == modules[0].offsets
