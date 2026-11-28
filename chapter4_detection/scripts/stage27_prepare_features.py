#!/usr/bin/env python3
"""Build Stage27 frozen-feature caches with unique spatially stratified negatives."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from chapter4_detection.scripts import stage26_prepare_features as base


class SpatialGenerator:
    """Drop-in subset of Generator implementing deterministic 4x8 stratum sampling."""
    def __init__(self, seed):
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def choice(self, candidates, size, replace=False):
        candidates = np.asarray(candidates)
        if replace or len(candidates) <= size:
            return self.rng.choice(candidates, size, replace=replace)
        # P2 is 384x384 for the frozen 1536 protocol. Candidate values are flat indices.
        y, x = candidates // 384, candidates % 384
        sid = np.minimum(y // 96, 3) * 8 + np.minimum(x // 48, 7)
        groups = {s: candidates[sid == s].copy() for s in np.unique(sid)}
        for values in groups.values(): self.rng.shuffle(values)
        offsets = Counter(); selected = []
        active = sorted(groups)
        while len(selected) < size and active:
            for s in list(active):
                i = offsets[s]
                if i >= len(groups[s]): active.remove(s); continue
                selected.append(groups[s][i]); offsets[s] += 1
                if len(selected) == size: break
        return np.asarray(selected, dtype=candidates.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--images-list', type=Path, required=True)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--negative-per-image', type=int, choices=(64, 256), required=True)
    args = p.parse_args()
    original = base.np.random.default_rng
    base.np.random.default_rng = lambda seed: SpatialGenerator(seed)
    old_argv = sys.argv
    sys.argv = ['stage26_prepare_features.py', '--weights', str(args.weights),
                '--images-list', str(args.images_list), '--fold', str(args.fold),
                '--output', str(args.output), '--seed', str(args.seed),
                '--negative-per-image', str(args.negative_per_image)]
    try:
        base.main()
    finally:
        base.np.random.default_rng = original; sys.argv = old_argv
    data = torch.load(args.output, map_location='cpu', weights_only=False)
    data['protocol'].update({'stage': 27, 'negative_sampling': 'unique_spatial_stratified_4x8',
                             'hard_e0_regions': 'excluded_pending_review',
                             'natural_sampling_required': True})
    torch.save(data, args.output)
    negatives = [m for m in data['meta'] if not m['positive']]
    unique = len({(m['image_index'], m['p2_y'], m['p2_x']) for m in negatives})
    per_image = Counter(m['image'] for m in negatives)
    audit = {'fold': args.fold, 'negative_per_image': args.negative_per_image,
             'samples': len(data['positive']), 'positive': int(data['positive'].sum()),
             'negative': len(negatives), 'unique_negative_locations': unique,
             'duplicates': len(negatives) - unique,
             'per_image_negative_min': min(per_image.values()),
             'per_image_negative_max': max(per_image.values()),
             'cache_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
             'protocol': data['protocol']}
    args.output.with_suffix('.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__': main()
