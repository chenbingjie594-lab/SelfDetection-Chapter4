#!/usr/bin/env python3
"""Stage27 FSPR training with natural sampling and a fixed optimizer budget."""
import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, TensorDataset

from chapter4_detection.models.fspr import FSPR


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou_loss(off, edge, target_off, target_edge):
    pred = torch.stack((off[:, 0] - edge[:, 0], off[:, 1] - edge[:, 1],
                        off[:, 0] + edge[:, 2], off[:, 1] + edge[:, 3]), 1)
    target = torch.stack((target_off[:, 0] - target_edge[:, 0], target_off[:, 1] - target_edge[:, 1],
                          target_off[:, 0] + target_edge[:, 2], target_off[:, 1] + target_edge[:, 3]), 1)
    lt, rb = torch.maximum(pred[:, :2], target[:, :2]), torch.minimum(pred[:, 2:], target[:, 2:])
    inter = (rb - lt).clamp_min(0).prod(1)
    pa = (pred[:, 2:] - pred[:, :2]).clamp_min(0).prod(1)
    ta = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(1)
    return (1 - inter / (pa + ta - inter + 1e-8)).mean()


class FixedBudgetNaturalSampler(Sampler):
    """Uniform sampling over the dataset; replacement is allowed only to equalize steps."""
    def __init__(self, size, samples, seed):
        self.size, self.samples, self.seed = size, samples, seed
        self.epoch = 0

    def __len__(self):
        return self.samples

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        full, rem = divmod(self.samples, self.size)
        chunks = [torch.randperm(self.size, generator=g) for _ in range(full)]
        if rem:
            chunks.append(torch.randperm(self.size, generator=g)[:rem])
        return iter(torch.cat(chunks).tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--variant', choices=('F1', 'F2'), required=True)
    p.add_argument('--experiment', choices=('R1', 'R2', 'R3'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-samples', type=int, required=True,
                   help='N256 cache size; defines the common 40-epoch optimizer budget')
    p.add_argument('--exclude-images-list', type=Path,
                   help='Inner-calibration images excluded from FSPR fitting')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data = torch.load(a.cache, map_location='cpu', weights_only=False)
    keep = torch.ones(len(data['positive']), dtype=torch.bool)
    excluded = set()
    if a.exclude_images_list:
        excluded = {Path(x).name for x in a.exclude_images_list.read_text().splitlines() if x.strip()}
        keep = torch.tensor([m['image'] not in excluded for m in data['meta']])
    positive = data['positive'].bool()[keep]
    dataset = TensorDataset(data['p2'].float()[keep], data['p3_up'].float()[keep], data['valid_mask'][keep],
                            data['local_index'][keep], data['heat'][keep], data['offset'].float()[keep],
                            data['edges'].float()[keep], positive)
    samples_per_epoch = int(a.reference_samples)
    sampler = FixedBudgetNaturalSampler(len(dataset), samples_per_epoch, a.seed)
    loader = DataLoader(dataset, batch_size=a.batch, sampler=sampler)
    model = FSPR(*data['channels'], variant=a.variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    exposure = torch.zeros(len(dataset), dtype=torch.int64)
    logs, batch_rows, start = [], [], time.time()
    for epoch in range(a.epochs):
        model.train(); sums = np.zeros(5); seen = pos_seen = 0
        # Reconstruct the exact sampler order once, then feed it through a loader.
        order = list(iter(sampler)); sampler.epoch -= 1
        exposure += torch.bincount(torch.tensor(order), minlength=len(dataset))
        for batch_id, (f2, f3, mask, loc, heat, target_off, target_edge, pos) in enumerate(loader):
            out = model(f2, f3, mask, origin=(0, 0)); ix = torch.arange(len(f2)); y, x = loc[:, 0], loc[:, 1]
            logits = out['logits'][ix, :, y, x]; off = out['offset'][ix, :, y, x]; edge = out['edges'][ix, :, y, x]
            prob = logits.sigmoid(); bce = F.binary_cross_entropy_with_logits(logits, heat, reduction='none')
            cls = (torch.where(heat > 0, (1 - prob).pow(2), prob.pow(2)) * bce).mean()
            pos = pos.bool()
            if pos.any():
                il = iou_loss(off[pos], edge[pos], target_off[pos], target_edge[pos])
                ol = F.smooth_l1_loss(off[pos], target_off[pos]); el = F.smooth_l1_loss(edge[pos], target_edge[pos])
            else:
                il = ol = el = off.sum() * 0
            loss = cls + il + ol + .5 * el
            if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            n = len(f2); seen += n; pos_seen += int(pos.sum())
            sums += np.array([float(loss), float(cls), float(il), float(ol), float(el)]) * n
            batch_rows.append({'epoch': epoch + 1, 'batch': batch_id, 'samples': n,
                               'positive': int(pos.sum()), 'negative': n - int(pos.sum())})
        row = {'epoch': epoch + 1, 'loss': sums[0] / seen, 'cls': sums[1] / seen,
               'iou': sums[2] / seen, 'offset': sums[3] / seen, 'edge': sums[4] / seen,
               'samples': seen, 'positive': pos_seen, 'negative': seen - pos_seen}
        logs.append(row); print(a.experiment, a.variant, *row.values(), flush=True)
    ckpt = {'variant': a.variant, 'experiment': a.experiment, 'state_dict': model.state_dict(),
            'channels': data['channels'], 'config': {'epochs': a.epochs, 'batch': a.batch,
            'optimizer': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'seed': a.seed,
            'sampling': 'uniform-natural', 'samples_per_epoch': samples_per_epoch,
            'optimizer_steps': a.epochs * math.ceil(samples_per_epoch / a.batch)},
            'cache_sha256': sha256(a.cache)}
    path = a.output / 'last.pt'; torch.save(ckpt, path)
    for name, rows in [('training_log.csv', logs), ('batch_sampling.csv', batch_rows)]:
        with (a.output / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0], lineterminator='\n'); w.writeheader(); w.writerows(rows)
    kept_meta = [m for m, use in zip(data['meta'], keep.tolist()) if use]
    gt_counts = Counter((m['image'], m['gt_id']) for m in kept_meta if m['positive'])
    report = {'experiment': a.experiment, 'variant': a.variant, 'samples': len(dataset),
              'positive': int(positive.sum()), 'negative': int((~positive).sum()),
              'natural_positive_fraction': float(positive.float().mean()),
              'inner_calibration_images_excluded': sorted(excluded),
              'samples_per_epoch': samples_per_epoch, 'optimizer_steps': ckpt['config']['optimizer_steps'],
              'exposure': {'min': int(exposure.min()), 'max': int(exposure.max()),
                           'mean': float(exposure.float().mean()),
                           'repeated_samples': int((exposure > a.epochs).sum())},
              'gt_participation_locations': {'min': min(gt_counts.values()), 'max': max(gt_counts.values()),
                                             'mean': float(np.mean(list(gt_counts.values())))},
              'parameters': sum(v.numel() for v in model.parameters()), 'final': logs[-1],
              'seconds': time.time() - start, 'checkpoint': str(path.resolve()),
              'checkpoint_sha256': sha256(path)}
    (a.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
