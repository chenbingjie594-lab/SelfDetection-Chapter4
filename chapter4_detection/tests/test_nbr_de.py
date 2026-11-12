import pytest
import torch

from chapter4_detection.models.nbr_de import NBRDE


@pytest.mark.parametrize("shape", [(1, 32, 13, 29), (2, 64, 16, 16)])
def test_shape_identity_and_backward(shape):
    module = NBRDE(shape[1])
    x = torch.randn(*shape, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)
    y.square().mean().backward()
    assert module.gamma.grad is not None and torch.isfinite(module.gamma.grad).all()


def test_reference_has_no_wraparound():
    module = NBRDE(16).eval()
    x = torch.zeros(1, 16, 15, 15)
    x[..., 0, 0] = 100
    with torch.no_grad():
        background = module._reference(x)
    assert torch.isfinite(background).all()
    assert background[..., -1, -1].abs().max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_finite_cuda():
    module = NBRDE(32).cuda().train()
    x = torch.randn(1, 32, 32, 32, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.float16):
        y = module(x)
    assert torch.isfinite(y).all()
    y.mean().backward()
