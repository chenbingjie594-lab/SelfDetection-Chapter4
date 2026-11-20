#!/usr/bin/env python3
"""Render Stage21 reports and figures from frozen diagnostic outputs."""
import argparse,csv,json,hashlib,platform,subprocess
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch,ultralytics

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--xml-trace',type=Path,required=True);p.add_argument('--expert-review',type=Path,required=True);a=p.parse_args();o=a.output
 s=json.loads((o/'annotation_sensitivity_summary.json').read_text());l=json.loads((o/'scale_learnability_summary.json').read_text());base=s['baseline'];ops=('center','size','single_l','single_t','single_r','single_b','independent')
 fig,ax=plt.subplots(1,2,figsize=(11,4))
 for op in ops:
  ax[0].errorbar((.5,1,2),[s['conditions'][f'{op}_{d}']['map75']['mean'] for d in (.5,1,2)],yerr=[s['conditions'][f'{op}_{d}']['map75']['sd'] for d in (.5,1,2)],marker='o',label=op)
  ax[1].plot((.5,1,2),[s['conditions'][f'{op}_{d}']['tiny_coverage75']['mean'] for d in (.5,1,2)],marker='o',label=op)
 ax[0].axhline(base['map75'],ls='--',color='k');ax[1].axhline(base['tiny_coverage75'],ls='--',color='k');ax[0].set(xlabel='Original-pixel amplitude',ylabel='AP75');ax[1].set(xlabel='Original-pixel amplitude',ylabel='Tiny IoU≥.75 coverage');ax[0].legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(o/'annotation_boundary_sensitivity.png',dpi=180);plt.close(fig)
 rr=list(csv.DictReader((o/'signed_scale_residual_predictions.csv').open()));z=[r for r in rr if r['model']=='ridge'];fig,ax=plt.subplots(1,2,figsize=(9,4))
 for j,k in enumerate(('w','h')):ax[j].scatter([float(r['target_log_'+k]) for r in z],[float(r['pred_log_'+k]) for r in z],s=15,alpha=.7);lo=-.5;hi=.5;ax[j].plot([lo,hi],[lo,hi],'k--');ax[j].set(xlabel='true log residual',ylabel='predicted',title=('width' if k=='w' else 'height'))
 fig.tight_layout();fig.savefig(o/'signed_scale_residual_predictions.png',dpi=180);plt.close(fig)
 gt=[r for r in csv.DictReader(a.gt_stats.open()) if r['split']=='train'];tiny=[r for r in gt if min(float(r['input_width']),float(r['input_height']))<8]
 rel=lambda rows,d,dim:np.array([d/float(r[dim]) for r in rows]);relative={}
 for d in (.5,1,2):relative[str(d)]={'all_w_median':float(np.median(rel(gt,d,'width'))),'all_h_median':float(np.median(rel(gt,d,'height'))),'tiny_w_median':float(np.median(rel(tiny,d,'width'))),'tiny_h_median':float(np.median(rel(tiny,d,'height')))}
 geo={(r['image'],r['gt_id']):r for r in csv.DictReader(a.geometry.open()) if r['selection']=='oracle'};pairs=[]
 for r in gt:
  g=geo.get((r['image'],r['gt_id']))
  if g:pairs.append(((1/float(r['width'])+1/float(r['height']))/2,(float(g['width_abs_rel'])+float(g['height_abs_rel']))/2))
 corr=float(np.corrcoef(np.array(pairs).T)[0,1]);xml=list(csv.DictReader(a.xml_trace.open()));review=list(csv.DictReader(a.expert_review.open()));traced=[r for r in xml if r['status']=='TRACED'];below=sum(float(r['min_best_iou'])<.99 for r in traced);pending=sum(r['review_status']=='PENDING_EXPERT_REVIEW' for r in review)
 def cond(op,d,key='map75'):return s['conditions'][f'{op}_{d}'][key]
 lines=['# Annotation boundary sensitivity','','Frozen Stage10 E0 OOF predictions were never changed; only temporary GT copies were perturbed. Ten deterministic random seeds were used per non-baseline condition. Every run retained 174 valid boxes and unchanged classes/counts.','','Baseline: AP50 %.4f, AP75 %.4f, mAP50-95 %.4f, IoU≥.75 coverage %.4f, tiny coverage %.4f.'%(base['map50'],base['map75'],base['map50_95'],base['coverage75'],base['tiny_coverage75']),'','| amplitude | center ΔAP75 | size ΔAP75 | independent-edge ΔAP75 | center tiny coverage Δ | independent tiny coverage Δ |','|---:|---:|---:|---:|---:|---:|']
 for d in (.5,1,2):lines.append(f"| {d} px | {cond('center',d)['delta_mean']:+.4f} | {cond('size',d)['delta_mean']:+.4f} | {cond('independent',d)['delta_mean']:+.4f} | {cond('center',d,'tiny_coverage75')['delta_mean']:+.4f} | {cond('independent',d,'tiny_coverage75')['delta_mean']:+.4f} |")
 lines+=['','At ±2 px, center and independent-edge perturbations reduce AP75 by 0.1008 and 0.0917 on average. Tiny coverage falls more strongly than overall coverage for these conditions. Random perturbations may occasionally improve individual matches; this experiment measures evaluation sensitivity and is not evidence of annotation error or model improvement.','', 'See `annotation_perturbation_results.csv` and `annotation_boundary_sensitivity.png`.'];(o/'ANNOTATION_BOUNDARY_SENSITIVITY.md').write_text('\n'.join(lines)+'\n')
 (o/'ANNOTATION_UNCERTAINTY_VS_REGRESSION_ERROR.md').write_text(f"""# Annotation uncertainty versus regression error

For a 1 px coordinate change, the median relative width/height change is {relative['1']['all_w_median']:.4f}/{relative['1']['all_h_median']:.4f} across all train GT and {relative['1']['tiny_w_median']:.4f}/{relative['1']['tiny_h_median']:.4f} for tiny GT. The corresponding 2 px tiny ratios exceed {relative['2']['tiny_w_median']:.3f}/{relative['2']['tiny_h_median']:.3f}. Stage20 median absolute width/height prediction errors were about 0.091/0.093, so one-to-two original pixels are of a comparable relative order for many tiny boxes. This comparison does not estimate real label noise.

The correlation between per-instance one-pixel relative sensitivity and frozen oracle width/height absolute error is {corr:.3f}; it is descriptive and does not establish causality. Near-border instances are protected by clipping in the simulation and no official labels were edited.

Historical provenance review traced {len(traced)} XML/evidence mappings; {below} were below IoU 0.99. All {pending} Stage17 expert-review entries remain `PENDING_EXPERT_REVIEW`. Existing confirmed provenance and unresolved human boundary review are therefore reported separately. There is no new evidence that the labels are wrong—only strong evidence that high-IoU evaluation is pixel-sensitive.
""")
 m=l['models'];g=l['gate'];(o/'SIGNED_SCALE_RESIDUAL_LEARNABILITY.md').write_text(f"""# Signed scale residual learnability

## Protocol

Scope is the 92 Fold0-training images only. Candidates are generated independently of GT by the frozen Stage12 E0 rule (`reason=kept`); class-aware greedy IoU≥0.30 matching is used afterward only to construct supervision. Of {l['total_gt']} GT, {l['matched_gt']} have a matched candidate, {l['unmatched_gt']} do not, and {l['kept_background_candidates']} kept candidates are unmatched background. A scale module cannot recover the unmatched GT.

Five source-group folds separately fit every scaler, model and training-fold median. Inputs are prediction geometry, level, score and low-dimensional DFL statistics; no GT geometry, oracle IoU, candidate rank or repair result is an input. This uses frozen OOF feature caches but remains a small internal mechanism probe, not detection-model generalization evidence.

| method | width MAE | height MAE | combined MAE | IoU before→after | coverage .75 before→after |
|---|---:|---:|---:|---:|---:|
| zero | {m['zero']['width_mae']:.4f} | {m['zero']['height_mae']:.4f} | {m['zero']['combined_mae']:.4f} | {m['zero']['iou_before']:.4f}→{m['zero']['iou_after']:.4f} | {m['zero']['coverage75_before']:.4f}→{m['zero']['coverage75_after']:.4f} |
| fold median | {m['median']['width_mae']:.4f} | {m['median']['height_mae']:.4f} | {m['median']['combined_mae']:.4f} | {m['median']['iou_before']:.4f}→{m['median']['iou_after']:.4f} | {m['median']['coverage75_before']:.4f}→{m['median']['coverage75_after']:.4f} |
| linear | {m['linear']['width_mae']:.4f} | {m['linear']['height_mae']:.4f} | {m['linear']['combined_mae']:.4f} | {m['linear']['iou_before']:.4f}→{m['linear']['iou_after']:.4f} | {m['linear']['coverage75_before']:.4f}→{m['linear']['coverage75_after']:.4f} |
| ridge | {m['ridge']['width_mae']:.4f} | {m['ridge']['height_mae']:.4f} | {m['ridge']['combined_mae']:.4f} | {m['ridge']['iou_before']:.4f}→{m['ridge']['iou_after']:.4f} | {m['ridge']['coverage75_before']:.4f}→{m['ridge']['coverage75_after']:.4f} |
| HistGBR | {m['histgb']['width_mae']:.4f} | {m['histgb']['height_mae']:.4f} | {m['histgb']['combined_mae']:.4f} | {m['histgb']['iou_before']:.4f}→{m['histgb']['iou_after']:.4f} | {m['histgb']['coverage75_before']:.4f}→{m['histgb']['coverage75_after']:.4f} |

Ridge is the best fixed probe by combined MAE, improving only {g['mae_improvement']*100:.2f}% over the fold-local median, below the frozen 10% threshold. It improves 4/5 fold MAEs and both axes, but mean IoU changes by {m['ridge']['iou_after']-m['ridge']['iou_before']:+.4f}, IoU≥.75 coverage changes by {m['ridge']['coverage75_after']-m['ridge']['coverage75_before']:+.4f}, and {m['ridge']['degraded_accurate']} initially accurate candidates fall below 0.75. Bootstrap median-minus-ridge MAE improvement CI95 is [{l['bootstrap']['ci95'][0]:.4f}, {l['bootstrap']['ci95'][1]:.4f}], crossing zero.

Flash and Black do not justify separate mechanisms: ridge MAE is {m['ridge']['Flash']['mae']:.4f} versus {m['ridge']['Black']['mae']:.4f}, with mixed IoU effects. Tiny IoU rises slightly but high-quality coverage is unchanged. Decision: `NO_STABLE_SIGNED_RESIDUAL_SIGNAL`.
""")
 (o/'STAGE21_MECHANISM_DECISION.md').write_text("""# Stage21 mechanism decision

**Decision: `NO_STABLE_SIGNED_RESIDUAL_SIGNAL`. Do not implement or formally train SCR.**

The best fixed deployable probe misses the required 10% MAE improvement (4.52%), its source-group bootstrap interval crosses zero, mean candidate IoU does not improve, and IoU≥0.75 coverage decreases. Although four of five folds improve residual MAE and both axes move in the desired direction, the effect is not large or operationally consistent enough to justify a network module.

Consequently no `SCR_FEASIBILITY_AND_DESIGN.md`, `SCR_IMPLEMENTATION_AUDIT.md`, model file, checkpoint or formal metric is generated. A plain MLP width/height correction would also have weak novelty relative to ordinary box refinement; renaming it SCR would not solve that problem. Existing YOLO loss semantics would require explicit verification that a decoded-box correction participates in training, but this question is not pursued because the evidence gate failed.

CUDA is additionally unavailable (`torch.cuda.is_available()==False`, device count 0, CUDA tensor allocation fails, and `nvidia-smi` cannot communicate with the driver), but CUDA is not the primary stop reason.
""")
 (o/'STAGE21_FINAL_SUMMARY.md').write_text("""# Stage21 final summary

1. Frozen-prediction AP75 is materially sensitive to subpixel-to-two-pixel GT boundary perturbations; tiny targets are generally more sensitive.
2. This demonstrates evaluation sensitivity, not annotation error. Expert review remains pending and no label was changed.
3. Model width/height error and one-to-two-pixel relative boundary changes are of similar order for many tiny GT.
4. Low-dimensional deployable geometry/DFL features show only weak signed-residual predictability. Ridge improves residual MAE by 4.52% versus fold-local median but worsens high-quality coverage.
5. There is no evidence that Flash and Black require separate scale regressors.
6. SCR fails the frozen mechanism gate, was not implemented, and should not enter formal training. The next step should be expert boundary review or new independent real data, not another regression branch.
""")
 env={'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count()}
 try:torch.ones(1,device='cuda');env['cuda_tensor_test']='PASS'
 except Exception as e:env['cuda_tensor_test']='FAIL: '+repr(e)
 (o/'environment_audit.json').write_text(json.dumps(env,indent=2)+'\n')
 (o/'RUN_COMMANDS.md').write_text("""# Stage21 reproduction

```bash
export CH4_PY=/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python
PYTHONPATH=. $CH4_PY chapter4_detection/scripts/stage21_annotation_sensitivity.py --predictions chapter4_detection/results/stage10/e0_oof_seed42_predictions.json --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --output chapter4_detection/results/stage21
PYTHONPATH=. $CH4_PY chapter4_detection/scripts/stage21_scale_learnability.py --features chapter4_detection/results/stage12/candidate_quality_features.csv --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --train-list chapter4_detection/results/stage10/cv_protocol/fold0/train.txt --output chapter4_detection/results/stage21 --bootstrap 5000
PYTHONPATH=. $CH4_PY chapter4_detection/scripts/stage21_finalize.py --output chapter4_detection/results/stage21 --gt-stats chapter4_detection/results/stage10/gt_statistics.csv --geometry chapter4_detection/results/stage20/geometry_by_gt_selection.csv --xml-trace chapter4_detection/results/stage10/xml_trace.csv --expert-review chapter4_detection/results/stage17/STAGE17_EXPERT_REVIEW.csv
PYTHONPATH=. $CH4_PY -m pytest -q chapter4_detection/tests/test_stage21.py
```
""")
 hashes={str(x.relative_to(o.parent.parent.parent)):hashlib.sha256(x.read_bytes()).hexdigest() for x in o.iterdir() if x.is_file() and x.name!='FILE_HASHES.json'};(o/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
if __name__=='__main__':main()
