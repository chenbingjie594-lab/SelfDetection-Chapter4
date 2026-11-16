from pathlib import Path

import pytest
import torch
from ultralytics import YOLO

from chapter4_detection.models.nbr_de import NBRDE, register_nbr_de


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


@pytest.mark.parametrize("mode", ["full", "no_reference", "no_gate"])
def test_mode_survives_checkpoint_reload(tmp_path: Path, mode: str):
    register_nbr_de()
    model = YOLO("chapter4_detection/models/yolo11s_nbr_de.yaml")
    module = next(m for m in model.model.modules() if isinstance(m, NBRDE))
    module.mode = mode
    checkpoint = tmp_path / f"{mode}.pt"
    model.save(checkpoint)
    reloaded = YOLO(str(checkpoint))
    loaded_module = next(m for m in reloaded.model.modules() if isinstance(m, NBRDE))
    assert loaded_module.mode == mode


def test_ablation_forward_paths_are_distinct():
    torch.manual_seed(7)
    module = NBRDE(16).eval()
    module.gamma.data.fill_(1.0)
    x = torch.randn(1, 16, 11, 17)
    outputs = []
    with torch.no_grad():
        for mode in ("full", "no_reference", "no_gate"):
            module.mode = mode
            outputs.append(module(x))
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])


def test_inference_interventions_preserve_checkpoint_parameters():
    torch.manual_seed(11)
    module = NBRDE(16).eval()
    module.gamma.data.fill_(0.5)
    x = torch.randn(1, 16, 11, 17)
    state_before = {k: v.clone() for k, v in module.state_dict().items()}
    with torch.no_grad():
        module.inference_intervention = "full"
        full = module(x)
        module.inference_intervention = "gate_bypass"
        no_gate = module(x)
        module.inference_intervention = "residual_bypass"
        no_residual = module(x)
    assert torch.equal(no_residual, x)
    assert not torch.equal(full, no_gate)
    assert all(torch.equal(v, state_before[k]) for k, v in module.state_dict().items())
    assert "inference_intervention" not in module.state_dict()


def test_gamma_zero_gradient_routing_then_branch_activation():
    torch.manual_seed(13)
    upstream = torch.nn.Conv2d(3, 16, 1)
    module = NBRDE(16).train()
    downstream = torch.nn.Conv2d(16, 4, 1)
    x = torch.randn(2, 3, 9, 11)
    loss = downstream(module(upstream(x))).square().mean()
    loss.backward()
    assert module.gamma.grad is not None and module.gamma.grad.abs().item() > 0
    assert upstream.weight.grad is not None and upstream.weight.grad.norm() > 0
    for name, parameter in module.named_parameters():
        if name.startswith("local"):
            assert parameter.grad is None  # inactive in full-reference mode
        elif name != "gamma":
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) == 0
    with torch.no_grad():
        module.gamma.fill_(0.1)
    upstream.zero_grad(); module.zero_grad(); downstream.zero_grad()
    downstream(module(upstream(x))).square().mean().backward()
    assert module.difference[0].weight.grad.norm() > 0
    assert module.gate.weight.grad.norm() > 0
    assert module.query.weight.grad.norm() > 0


def test_identity_train_skips_branch_and_bn_updates():
    module = NBRDE(16, mode="identity_train").train()
    before = module.difference[1].running_mean.clone()
    x = torch.randn(2, 16, 9, 11, requires_grad=True)
    y = module(x)
    assert y is x
    y.square().mean().backward()
    assert x.grad is not None and x.grad.norm() > 0
    assert all(p.grad is None for p in module.parameters())
    assert torch.equal(before, module.difference[1].running_mean)


@pytest.mark.parametrize(
    ("yaml_path", "expected_mode"),
    [
        ("chapter4_detection/models/yolo11s_nbr_de_no_reference.yaml", "no_reference"),
        ("chapter4_detection/models/yolo11s_nbr_de_no_gate.yaml", "no_gate"),
    ],
)
def test_ablation_yaml_encodes_mode_and_survives_reload(
    tmp_path: Path, yaml_path: str, expected_mode: str
):
    """The trainer rebuilds from YAML, so runtime-only assignments are unsafe."""
    register_nbr_de()
    model = YOLO(yaml_path)
    module = next(m for m in model.model.modules() if isinstance(m, NBRDE))
    assert module.mode == expected_mode
    checkpoint = tmp_path / f"{expected_mode}.pt"
    model.save(checkpoint)
    reloaded = YOLO(str(checkpoint))
    loaded_module = next(m for m in reloaded.model.modules() if isinstance(m, NBRDE))
    assert loaded_module.mode == expected_mode
