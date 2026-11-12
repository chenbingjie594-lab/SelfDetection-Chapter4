"""Normal Background Reference and Defect Enhancement (NBR-DE)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NBRDE(nn.Module):
    """Center-excluded sparse local reference and gated residual enhancement."""

    def __init__(self, channels: int = 128, mode: str = "full", distances=(2, 6)):
        super().__init__()
        if mode not in {"full", "no_reference", "no_gate"}:
            raise ValueError(f"Unknown NBR-DE mode: {mode}")
        self.channels, self.mode = channels, mode
        hidden = max(16, channels // 4)
        self.offsets = tuple((dy * d, dx * d) for d in distances for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        self.query = nn.Conv2d(channels, hidden, 1, bias=False)
        self.key = nn.Conv2d(channels, hidden, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
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

    def _reference(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "no_reference":
            return self.local(x)
        q, k, v = self.query(x), self.key(x), self.value(x)
        pad = max(max(abs(y), abs(z)) for y, z in self.offsets)
        valid = x.new_ones((x.shape[0], 1, *x.shape[-2:]))
        logits = []
        scale = q.shape[1] ** -0.5
        for dy, dx in self.offsets:
            ks = self._sample(k, dy, dx, pad)
            mask = self._sample(valid, dy, dx, pad).bool()
            score = (q * ks).sum(1, keepdim=True) * scale
            logits.append(score.masked_fill(~mask, torch.finfo(score.dtype).min))
        weights = torch.softmax(torch.cat(logits, 1).float(), 1).to(x.dtype)
        background = torch.zeros_like(v)
        for index, (dy, dx) in enumerate(self.offsets):
            background = background + weights[:, index:index + 1] * self._sample(v, dy, dx, pad)
        return background

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        background = self._reference(x)
        residual = x - background
        difference = self.difference(torch.cat((residual, residual.abs()), 1))
        gate = torch.ones_like(difference) if self.mode == "no_gate" else torch.sigmoid(self.gate(torch.cat((x, difference, background), 1)))
        output = x + self.gamma * gate * difference
        if self.export_features:
            self.last_features = {"F": x.detach(), "B": background.detach(), "D": difference.detach(), "G": gate.detach()}
        return output


def register_nbr_de() -> None:
    """Register with Ultralytics YAML parser without editing site-packages."""
    import ultralytics.nn.tasks as tasks
    tasks.NBRDE = NBRDE
