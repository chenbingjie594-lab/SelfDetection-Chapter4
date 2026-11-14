"""Reliability-guided Normal Background Reference and Defect Enhancement."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NBRDER(nn.Module):
    """Blend sparse-attention and local references using normalized inconsistency."""

    def __init__(self, channels: int = 128, distances=(2, 6), eps: float = 1e-6):
        super().__init__()
        self.channels, self.eps = channels, eps
        hidden = max(16, channels // 4)
        self.offsets = tuple((dy * d, dx * d) for d in distances for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        self.query = nn.Conv2d(channels, hidden, 1, bias=False)
        self.key = nn.Conv2d(channels, hidden, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.reliability_bias = nn.Parameter(torch.zeros(1))
        self.reliability_slope_raw = nn.Parameter(torch.zeros(1))
        self.difference = nn.Sequential(nn.Conv2d(2 * channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.SiLU())
        self.gate = nn.Conv2d(3 * channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.export_features = False
        self.last_features: dict[str, torch.Tensor] = {}

    @staticmethod
    def _sample(x: torch.Tensor, dy: int, dx: int, pad: int) -> torch.Tensor:
        h, w = x.shape[-2:]
        padded = F.pad(x, (pad, pad, pad, pad))
        return padded[..., pad + dy:pad + dy + h, pad + dx:pad + dx + w]

    def reliability(self, inconsistency: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.reliability_bias - F.softplus(self.reliability_slope_raw) * inconsistency)

    def references(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        q, k, v = self.query(x), self.key(x), self.value(x)
        pad = max(max(abs(y), abs(z)) for y, z in self.offsets)
        valid = x.new_ones((x.shape[0], 1, *x.shape[-2:]))
        sampled_v, logits = [], []
        scale = q.shape[1] ** -0.5
        for dy, dx in self.offsets:
            ks = self._sample(k, dy, dx, pad)
            vs = self._sample(v, dy, dx, pad)
            mask = self._sample(valid, dy, dx, pad).bool()
            score = (q * ks).sum(1, keepdim=True) * scale
            logits.append(score.masked_fill(~mask, torch.finfo(score.dtype).min))
            sampled_v.append(vs)
        weights = torch.softmax(torch.cat(logits, 1).float(), 1).to(x.dtype)
        attention = torch.zeros_like(v)
        for index, vs in enumerate(sampled_v):
            attention = attention + weights[:, index:index + 1] * vs
        numerator = torch.zeros_like(weights[:, :1])
        denominator = torch.zeros_like(weights[:, :1])
        for index, vs in enumerate(sampled_v):
            weight = weights[:, index:index + 1]
            numerator = numerator + weight * (vs - attention).square().mean(1, keepdim=True)
            denominator = denominator + weight * vs.square().mean(1, keepdim=True)
        inconsistency = numerator / (denominator + self.eps)
        trust = self.reliability(inconsistency)
        local = self.local(x)
        blended = trust * attention + (1.0 - trust) * local
        return attention, local, inconsistency, trust, blended, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention, local, inconsistency, trust, background, weights = self.references(x)
        residual = x - background
        difference = self.difference(torch.cat((residual, residual.abs()), 1))
        gate = torch.sigmoid(self.gate(torch.cat((x, difference, background), 1)))
        contribution = self.gamma * gate * difference
        output = x + contribution
        if self.export_features:
            self.last_features = {"F": x.detach(), "B_att": attention.detach(), "B_local": local.detach(),
                                  "U": inconsistency.detach(), "r": trust.detach(), "B": background.detach(),
                                  "D": difference.detach(), "G": gate.detach(), "contribution": contribution.detach(),
                                  "attention_weights": weights.detach()}
        return output


def register_nbr_de_r() -> None:
    """Register with the Ultralytics YAML parser without changing site-packages."""
    import ultralytics.nn.tasks as tasks
    tasks.NBRDER = NBRDER
