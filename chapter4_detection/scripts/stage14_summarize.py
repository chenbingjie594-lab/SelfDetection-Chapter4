#!/usr/bin/env python3
"""Summarize the frozen Stage14 per-positive TAL/DFL geometry trace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EDGES = ("l", "t", "r", "b")
GROUP = ["epoch", "assignment_call", "batch_index", "gt_index"]


def weighted_rate(mask: pd.Series, weight: pd.Series) -> float:
    return float(weight[mask].sum() / weight.sum()) if weight.sum() else 0.0


def bootstrap_ci(values: np.ndarray, seed: int = 42, n: int = 10000) -> list[float]:
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (n, len(values)), replace=True).mean(1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.trace)
    required = set(GROUP + ["inside_original", "target_score", "representable_iou_max"])
    if missing := required - set(d):
        raise RuntimeError(f"missing required columns: {sorted(missing)}")
    d["outside"] = ~d.inside_original.astype(bool)
    flag_cols = [f"clipped_{x}_flag" for x in EDGES]
    d["clipped_any"] = d[flag_cols].astype(bool).any(axis=1)
    d["negative_edges"] = (d[[f"signed_{x}" for x in EDGES]] < 0).sum(axis=1)
    d["upper_clipped_edges"] = (d[[f"signed_{x}" for x in EDGES]] > 14.99).sum(axis=1)
    d["clipped_edges"] = d.negative_edges + d.upper_clipped_edges

    rows = []
    for epoch, g in d.groupby("epoch", sort=True):
        rows.append({
            "epoch": int(epoch), "positive_count": len(g), "outside_count": int(g.outside.sum()),
            "outside_count_ratio": float(g.outside.mean()),
            "outside_weight_ratio": weighted_rate(g.outside, g.target_score),
            "clipped_positive_ratio": float(g.clipped_any.mean()),
            "representable_iou_lt_050_ratio": float((g.representable_iou_max < .5).mean()),
            "representable_iou_lt_075_ratio": float((g.representable_iou_max < .75).mean()),
        })
    epoch = pd.DataFrame(rows)
    epoch.to_csv(args.output / "geometry_by_epoch.csv", index=False)

    gt_rows = []
    paired_rows = []
    for key, g in d.groupby(GROUP, sort=False):
        inside, outside = g[~g.outside], g[g.outside]
        base = dict(zip(GROUP, key))
        base.update(class_id=int(g.class_id.iloc[0]), gt_width=float(g.gt_width.iloc[0]),
                    gt_height=float(g.gt_height.iloc[0]), is_tiny=bool(g.is_tiny.iloc[0]),
                    positive_count=len(g), outside_count=len(outside), outside_weight=float(outside.target_score.sum()),
                    total_weight=float(g.target_score.sum()), all_outside=bool(len(inside) == 0),
                    has_outside=bool(len(outside) > 0), representable_lt050=bool((g.representable_iou_max < .5).any()),
                    representable_lt075=bool((g.representable_iou_max < .75).any()))
        gt_rows.append(base)
        if len(inside) and len(outside):
            p = dict(base)
            for col in ["pred_iou_clamped", "target_score", "dfl_loss_unweighted", "dfl_loss_weighted",
                        "ciou_loss_weighted", "center_error_px", "width_error_px", "height_error_px",
                        "representable_iou_max"]:
                p[f"inside_{col}"] = float(inside[col].mean())
                p[f"outside_{col}"] = float(outside[col].mean())
                p[f"delta_outside_minus_inside_{col}"] = p[f"outside_{col}"] - p[f"inside_{col}"]
            paired_rows.append(p)
    gt = pd.DataFrame(gt_rows)
    paired = pd.DataFrame(paired_rows)
    gt.to_csv(args.output / "geometry_by_augmented_gt.csv", index=False)
    paired.to_csv(args.output / "paired_inside_outside_by_gt.csv", index=False)

    pair_summary = {}
    for col in ["pred_iou_clamped", "target_score", "dfl_loss_unweighted", "dfl_loss_weighted",
                "ciou_loss_weighted", "center_error_px", "representable_iou_max"]:
        vals = paired[f"delta_outside_minus_inside_{col}"].to_numpy()
        pair_summary[col] = {"paired_gt_count": len(vals), "inside_mean": float(paired[f"inside_{col}"].mean()),
                             "outside_mean": float(paired[f"outside_{col}"].mean()),
                             "mean_delta_outside_minus_inside": float(vals.mean()),
                             "mean_delta_bootstrap_95ci": bootstrap_ci(vals)}

    summary = {
        "trace_rows": len(d), "trace_epochs": sorted(int(x) for x in d.epoch.unique()),
        "outside_positive_count": int(d.outside.sum()), "outside_positive_ratio": float(d.outside.mean()),
        "outside_weight_ratio": weighted_rate(d.outside, d.target_score),
        "outside_negative_distance_ratio": float((d.loc[d.outside, "negative_edges"] > 0).mean()),
        "outside_clipped_ratio": float(d.loc[d.outside, "clipped_any"].mean()),
        "upper_clipped_positive_ratio": float((d.upper_clipped_edges > 0).mean()),
        "representable_iou_lt_050_ratio": float((d.representable_iou_max < .5).mean()),
        "representable_iou_lt_075_ratio": float((d.representable_iou_max < .75).mean()),
        "outside_representable_iou_lt_050_ratio": float((d.loc[d.outside, "representable_iou_max"] < .5).mean()),
        "outside_representable_iou_lt_075_ratio": float((d.loc[d.outside, "representable_iou_max"] < .75).mean()),
        "weighted_representable_iou_lt_050_ratio": weighted_rate(d.representable_iou_max < .5, d.target_score),
        "weighted_representable_iou_lt_075_ratio": weighted_rate(d.representable_iou_max < .75, d.target_score),
        "augmented_gt_count": len(gt), "gt_with_outside_ratio": float(gt.has_outside.mean()),
        "gt_all_outside_ratio": float(gt.all_outside.mean()), "tiny_gt_all_outside_ratio": float(gt.loc[gt.is_tiny, "all_outside"].mean()),
        "paired_augmented_gt_count": len(paired), "paired": pair_summary,
        "edge_negative_counts": {x: int((d[f"signed_{x}"] < 0).sum()) for x in EDGES},
        "edge_upper_clip_counts": {x: int((d[f"signed_{x}"] > 14.99).sum()) for x in EDGES},
        "limitations": ["Rows identify augmented GT by epoch/assignment_call/batch_index/gt_index.",
                        "Source image ID is not recoverable for mosaic-composed batches and was not inferred."],
    }
    (args.output / "geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(d.anchor_x[d.outside], d.anchor_y[d.outside], s=3, alpha=.15, label="outside")
    axes[0].scatter(d.anchor_x[~d.outside], d.anchor_y[~d.outside], s=3, alpha=.15, label="inside")
    axes[0].set(xlabel="anchor x (network-input px)", ylabel="anchor y (network-input px)", title="Assigned anchor locations")
    axes[0].legend(markerscale=4)
    axes[1].hist([d.loc[~d.outside, "representable_iou_max"], d.loc[d.outside, "representable_iou_max"]],
                 bins=np.linspace(0, 1, 41), label=["inside", "outside"], alpha=.8)
    axes[1].axvline(.5, color="k", ls="--", lw=1); axes[1].axvline(.75, color="k", ls=":", lw=1)
    axes[1].set(xlabel="theoretical maximum IoU", ylabel="positive anchors", title="DFL geometric representability")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(args.output / "outside_positive_geometry.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(epoch.epoch, 100 * epoch.outside_count_ratio, "o-", label="count ratio")
    ax.plot(epoch.epoch, 100 * epoch.outside_weight_ratio, "s-", label="loss-weight ratio")
    ax.set(xlabel="epoch", ylabel="outside positives (%)", title="Outside-positive training dynamics")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(args.output / "geometry_dynamics.png", dpi=180); plt.close(fig)


if __name__ == "__main__":
    main()
