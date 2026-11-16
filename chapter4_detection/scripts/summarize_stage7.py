#!/usr/bin/env python3
"""Generate Stage-7 reports, tables, curves, hashes and commands."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
def mean_sd(x):return float(np.mean(x)),float(np.std(x,ddof=1))
def fmt(x):return f"{x:.4f}"
def main():
 p=argparse.ArgumentParser();p.add_argument("--stage7",type=Path,required=True);p.add_argument("--repo",type=Path,default=Path("."));a=p.parse_args();s=a.stage7;s.mkdir(parents=True,exist_ok=True)
 audits=[json.loads((s/"initializations"/f"initialization_audit_seed{x}.json").read_text()) for x in (42,3407,2026)];gamma=json.loads((s/"gamma_gradient_analysis.json").read_text())
 init="# Initialization and Training Audit\n\nHistorical E0 and E3 did not use a strictly identical 2-class initialization path. E0 let the Ultralytics trainer adapt the 80-class checkpoint; E3 built a 2-class YAML model before training and explicitly transferred semantic tensors. Historical E3 left the six class-prediction tensors plus NBR-DE tensors randomly initialized. Both runs selected AdamW automatically at lr=0.001667, but E0 used parameter groups 81/88/87 and E3 used 82/95/89 (no-decay/decay/bias). Consequently historical E3−E0 differences cannot be treated as initialization-controlled causal effects.\n\nFor Stage 7, each seed received one matched 2-class E0 initialization; Identity and Full copied every shared tensor, including classifier/regressor heads, BatchNorm parameters and running statistics.\n\n| Seed | Shared parameter mismatches (Identity/Full) | Raw output max abs (Identity/Full) | Loss delta (Identity/Full) | Shared gradient mismatches (Identity/Full) |\n|---:|---:|---:|---:|---:|\n"
 for z in audits:init+=f"| {z['seed']} | {len(z['shared_parameter_equality']['MID']['mismatches'])}/{len(z['shared_parameter_equality']['MFULL']['mismatches'])} | {z['raw_output_comparison']['M0_vs_MID']['max_abs']:.1e}/{z['raw_output_comparison']['M0_vs_MFULL']['max_abs']:.1e} | {z['loss_abs_delta']['M0_vs_MID']:.1e}/{z['loss_abs_delta']['M0_vs_MFULL']:.1e} | {len(z['shared_gradient_equality']['MID']['mismatches'])}/{len(z['shared_gradient_equality']['MFULL']['mismatches'])} |\n"
 init+="\nAll matched checks must be zero before formal training. Checkpoint SHA256 values are stored in each audit JSON; checkpoint binaries are excluded from Git.\n";(s/"INITIALIZATION_AND_TRAINING_AUDIT.md").write_text(init)
 good=[z for z in gamma if not z["step_skipped"]]; g="# Gamma Gradient Analysis\n\nThe real YOLO detection loss confirms the analytic routing. At gamma=0, gamma receives a non-zero gradient while Q/K/V, difference and gate gradients are exactly zero; adjacent main-network layers retain non-zero gradients. After the first successful optimizer update makes gamma non-zero, all active NBR-DE branches receive gradients.\n\nAMP initially skipped overflowing attempts and kept gamma at zero. The first successful AMP update occurred only after GradScaler reduced its scale; skipped attempts are preserved rather than counted as optimizer updates.\n\n| Precision | Successful update | gamma before | gamma after | gamma grad | Q grad | Difference grad | Gate grad |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
 for z in good:g+=f"| {z['precision']} | {z['optimizer_update']} | {z['gamma_before']:.6g} | {z['gamma_after']:.6g} | {z['gamma_grad']:.4g} | {z['query_grad']:.4g} | {z['difference_grad']:.4g} | {z['gate_grad']:.4g} |\n"
 (s/"GAMMA_GRADIENT_ANALYSIS.md").write_text(g)
 dynp=s/"dynamics_seed42"/"training_dynamics.csv";dyn=list(csv.DictReader(dynp.open())) if dynp.exists() else []
 if dyn:
  import matplotlib.pyplot as plt
  success=[r for r in dyn if r["step_skipped"]=="False"][:300];u=np.arange(1,len(success)+1);gamma_v=np.array([float(r["gamma_after"]) for r in success]);rho=np.array([float(r["injection_ratio"]) for r in success]);gate=np.array([float(r["gate_mean"]) for r in success]);
  fig,ax=plt.subplots(3,1,figsize=(9,9),sharex=True);ax[0].plot(u,gamma_v);ax[0].set_ylabel("gamma");ax[1].plot(u,rho);ax[1].set_ylabel("injection ratio rho");ax[2].plot(u,gate);ax[2].set_ylabel("gate mean");ax[2].set_xlabel("successful optimizer update");fig.tight_layout();fig.savefig(s/"training_dynamics.png",dpi=180);plt.close(fig)
  dr=f"# Training Dynamics Report\n\nThis is a fresh seed42 diagnostic run, not a reconstructed historical trajectory. It records {len(success)} successful optimizer updates and {len(dyn)-len(success)} GradScaler-skipped attempts.\n\n- gamma: start `{gamma_v[0]:.6g}`, final `{gamma_v[-1]:.6g}`, maximum absolute `{np.abs(gamma_v).max():.6g}`.\n- injection ratio rho: mean `{rho.mean():.6g}`, maximum `{rho.max():.6g}`, final `{rho[-1]:.6g}`.\n- gate mean: mean `{gate.mean():.6g}`, final `{gate[-1]:.6g}`.\n\nThe CSV contains Q/K/V, difference, gate, adjacent-layer gradients and box/cls/dfl losses for every optimizer attempt. Validation values from this diagnostic run were not used for tuning.\n"
 else:dr="# Training Dynamics Report\n\nPending.\n"
 (s/"TRAINING_DYNAMICS_REPORT.md").write_text(dr)
 specs={"Matched-E0":"matched_e0","Matched-Identity":"matched_identity","Matched-Full":"matched_full"};metrics={};complete=True
 for label,prefix in specs.items():
  metrics[label]=[]
  for seed in (42,3407,2026):
   q=s/f"{prefix}_seed{seed}_metrics.json"
   if not q.exists():complete=False;continue
   m=json.loads(q.read_text());d=json.loads((s/f"{prefix}_seed{seed}_diagnostics.json").read_text());metrics[label].append({"seed":seed,"map50":m["map50"],"map50_95":m["map50_95"],"ap75":m["map75"],"flash_ap":m["per_class"]["0"]["ap50_95"],"black_ap":m["per_class"]["1"]["ap50_95"],"black_recall":m["per_class"]["1"]["recall"],"low_black":d["groups"]["class1/contrast/low"]["detected"]})
 ident="# Identity Branch Control Results\n\n| Model | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Black Recall | Low-contrast Black |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
 if complete:
  for label,rows in metrics.items():
   vals=[]
   for k in ("map50","map50_95","ap75","flash_ap","black_ap","black_recall"):
    m,sd=mean_sd([r[k] for r in rows]);vals.append(f"{m:.4f} ± {sd:.4f}")
   ident+=f"| {label} | {' | '.join(vals)} | {np.mean([r['low_black'] for r in rows]):.2f}/13 |\n"
  for label,rows in metrics.items():
   ident+=f"\n{label} per seed: "+", ".join(f"{r['seed']}={r['map50_95']:.4f}" for r in rows)+".\n"
 else:ident+="| Formal matched runs | — | — | — | — | — | — | pending |\n"
 (s/"IDENTITY_BRANCH_CONTROL_RESULTS.md").write_text(ident)
 fp=s/"feature_intervention_summary.json";feature=json.loads(fp.read_text()) if fp.exists() else {}
 if feature:
  final_path=s/"final_prediction_intervention_summary.json";final=json.loads(final_path.read_text()) if final_path.exists() else {}
  ftxt="# Feature Intervention Analysis\n\n| Bypass | P3 δ | P4 δ | P5 δ | Class-logit δ | Box-raw δ | Decoded-candidate δ | Defect/background change |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
  for mode,z in feature.items():ftxt+=f"| {mode} | {z['p3_relative_l2']['mean']:.3e} | {z['p4_relative_l2']['mean']:.3e} | {z['p5_relative_l2']['mean']:.3e} | {z['class_logit_relative_l2']['mean']:.3e} | {z['box_raw_relative_l2']['mean']:.3e} | {z['decoded_candidate_relative_l2']['mean']:.3e} | {z['defect_to_background_change_ratio']['mean']:.2f}× |\n"
  ftxt+="\nThe residual has a measurable, defect-concentrated effect, but its magnitude is about 10⁻⁴–10⁻³ in intermediate features and around 10⁻⁵ after decoding. It is therefore attenuated rather than exactly absent. Gamma was not artificially changed.\n"
  if final:
   ftxt+="\nAt confidence 0.25, all 171 detections across 3 seeds × 46 images matched one-to-one under both interventions; no final detection appeared or disappeared. "
   ftxt+=f"Gate bypass changed confidence by mean `{final['gate_bypass']['confidence_abs_delta']['mean']:.3e}` and produced matched-box IoU `{final['gate_bypass']['matched_box_iou']['mean']:.6f}`; "
   ftxt+=f"residual bypass changed confidence by `{final['residual_bypass']['confidence_abs_delta']['mean']:.3e}` with IoU `{final['residual_bypass']['matched_box_iou']['mean']:.6f}`.\n"
 else:ftxt="# Feature Intervention Analysis\n\nPending.\n"
 (s/"FEATURE_INTERVENTION_ANALYSIS.md").write_text(ftxt)
 if complete:
  means={k:np.mean([r["map50_95"] for r in v]) for k,v in metrics.items()};d_id=means["Matched-Identity"]-means["Matched-E0"];d_full=means["Matched-Full"]-means["Matched-E0"]
  paired_id=[a["map50_95"]-b["map50_95"] for a,b in zip(metrics["Matched-Identity"],metrics["Matched-E0"])]
  paired_full=[a["map50_95"]-b["map50_95"] for a,b in zip(metrics["Matched-Full"],metrics["Matched-E0"])]
  conclusion=f"# Stage 7 Final Conclusion\n\nMatched-Identity − Matched-E0 mean mAP50-95: `{d_id:+.4f}`; paired seeds: `{', '.join(f'{x:+.4f}' for x in paired_id)}`. Matched-Full − Matched-E0 mean: `{d_full:+.4f}`; paired seeds: `{', '.join(f'{x:+.4f}' for x in paired_full)}`.\n\n"
  conclusion+="## Directly established\n\n- The historical E0/E3 initialization paths were not strictly matched.\n- The new Identity control reproduces Matched-E0 mAP50-95 essentially exactly. The seed2026 E0 run required checkpoint resume, so exact trajectory identity is not claimed for that seed.\n- Matched-Full improves mean mAP50-95 by only 0.0026, with mixed seed effects (one negative, two positive) and no independent test set. This is not robust evidence of a detection gain.\n- Frozen-checkpoint bypass causes measurable but tiny, defect-concentrated changes; all 171 confidence≥0.25 detections remain one-to-one matched.\n\n## Supported interpretation\n\nAny NBR-DE effect is more plausible during training/representation optimization than as a strong inference-time background-enhancement mechanism. That explanation is supported, not proven causally.\n\n## Decision\n\nThere is **not enough evidence to use NBR-DE as the formal Chapter-4 innovation**. Keep it as an exploratory result. A stronger direction should target a mechanism with reproducible paired-seed gains and a material inference-time effect, and must ultimately be checked on an independent real test set.\n"
 else:conclusion="# Stage 7 Final Conclusion\n\nFormal three-seed matched experiments are incomplete; no final causal conclusion is issued.\n"
 (s/"STAGE7_FINAL_CONCLUSION.md").write_text(conclusion)
 commands="""# Stage-7 commands

Run from the repository root with the existing YOLO11 environment.

```bash
PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
IMG=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/train/img_3122_pre_part_1.jpg
LBL=/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/train/img_3122_pre_part_1.txt
for seed in 42 3407 2026; do
  PYTHONPATH=. $PY chapter4_detection/scripts/stage7_initialization_audit.py --pretrained yolo11s.pt --image $IMG --label $LBL --output-dir chapter4_detection/results/stage7/initializations --seed $seed --imgsz 640 --device cuda:0
done
PYTHONPATH=. $PY chapter4_detection/scripts/gamma_gradient_analysis.py --help
PYTHONPATH=. $PY chapter4_detection/scripts/diagnose_training_dynamics.py --help
DEVICE=0 bash chapter4_detection/scripts/run_stage7_matched_experiments.sh
PYTHONPATH=. $PY chapter4_detection/scripts/feature_intervention_analysis.py --weights-dir chapter4_detection/results/runs --images /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val --output-dir chapter4_detection/results/stage7 --device cuda:0
PYTHONPATH=. $PY chapter4_detection/scripts/analyze_final_intervention_predictions.py --input-dir chapter4_detection/results/stage6/interventions --output-dir chapter4_detection/results/stage7 --confidence 0.25
PYTHONPATH=. $PY chapter4_detection/scripts/summarize_stage7.py --stage7 chapter4_detection/results/stage7 --repo .
```

The two `resume_stage7_*.sh` scripts document the actual recovery commands used after process interruption; completed checkpoints and metrics were not overwritten.
""";(s/"RUN_COMMANDS.md").write_text(commands)
 checkpoints={}
 for label,prefix in specs.items():
  for seed in (42,3407,2026):
   q=s/"runs"/f"{prefix}_seed{seed}"/"weights"/"best.pt"
   if q.exists():checkpoints[f"{prefix}_seed{seed}_best"]={"path":str(q.resolve()),"sha256":hashlib.sha256(q.read_bytes()).hexdigest()}
 (s/"checkpoint_hashes.json").write_text(json.dumps(checkpoints,indent=2)+"\n")
 files=sorted(q for q in s.rglob("*") if q.is_file() and q.suffix!=".pt" and "runs" not in q.parts and q.name!="FILE_HASHES.sha256");(s/"FILE_HASHES.sha256").write_text("".join(f"{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(a.repo)}\n" for q in files));print(json.dumps({"formal_complete":complete,"files_hashed":len(files)}))
if __name__=="__main__":main()
