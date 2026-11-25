"""Training-only hard-background discrimination losses for the frozen MBRD adapter."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class HBDProjection(nn.Module):
    def __init__(self, channels: int, dim: int = 32):
        super().__init__()
        self.proj = nn.Linear(channels, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)

def _mean(x: torch.Tensor, weight: torch.Tensor | None = None):
    if weight is None:
        return F.normalize(x.mean(0), dim=0).detach()
    weight = weight.to(x).clamp_min(0)
    return F.normalize((x * weight[:, None]).sum(0) / weight.sum().clamp_min(1e-8), dim=0).detach()

def hbd_loss(u: torch.Tensor, sample_type: torch.Tensor, target_class: torch.Tensor,
             weight: torch.Tensor, margin: float = .1, tau: float = .1,
             class_conditional: bool = True):
    """Prototype margin loss. sample_type: 0 positive, 1 easy bg, 2 weak hard bg."""
    pos = sample_type == 0
    hard = sample_type == 2
    losses, valid = [], []
    classes = (0, 1) if class_conditional else (-1,)
    for cls in classes:
        pm = pos if cls < 0 else pos & (target_class == cls)
        nm = hard if cls < 0 else hard & (target_class == cls)
        if int(pm.sum()) < 2 or int(nm.sum()) < 1:
            continue
        p = _mean(u[pm])
        n = _mean(u[nm], weight[nm])
        lp = F.softplus(((u[pm] @ n) - (u[pm] @ p) + margin) / tau).mean()
        ln_each = F.softplus(((u[nm] @ p) - (u[nm] @ n) + margin) / tau)
        ln = (ln_each * weight[nm]).sum() / weight[nm].sum().clamp_min(1e-8)
        losses.extend((lp, ln)); valid.append({'class': cls, 'positive': int(pm.sum()), 'hard': int(nm.sum())})
    return (torch.stack(losses).mean() if losses else u.sum() * 0), valid
