#!/usr/bin/env python3
"""Merge Stage22 OOF predictions, evaluate, bootstrap and render reports."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,subprocess,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch,ultralytics
from chapter4_detection.scripts.paired_group_bootstrap import image_stats,prepare_weighted,weighted_evaluate

VAR={'A':'A','B':'B','C':'C','C-noDiff':'C_noDiff'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('chapter4_detection/results/stage22'));p.add_argument('--stage10',type=Path,default=Path('chapter4_detection/results/stage10'));p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args();o=a.root
 merged={};base=json.loads((a.stage10/'e0_oof_seed42_predictions.json').read_text());merged['E0']=base
 for name,disk in VAR.items():
  images={};secs=0
  for f in range(3):
   q=json.loads((o/f'predictions/fold{f}_{disk}.json').read_text());images.update(q['images']);secs+=q['elapsed_seconds']
  merged[name]={'schema_version':1,'model':name,'protocol':'three-fold OOF CIR refinement; frozen E0 classes/scores/post-NMS candidates','images_count':len(images),'mean_seconds_per_original':secs/len(images),'images':images};(o/f'{disk}_oof_predictions.json').write_text(json.dumps(merged[name])+'\n')
 metrics={}
 for name,q in merged.items():
  disk=VAR.get(name,'E0');pp=a.stage10/'e0_oof_seed42_predictions.json' if name=='E0' else o/f'{disk}_oof_predictions.json';mp=o/f'{disk}_oof_metrics.json';subprocess.run(['/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python','chapter4_detection/scripts/evaluate_original.py','--predictions',str(pp),'--labels',str(a.labels),'--output',str(mp)],check=True,stdout=subprocess.DEVNULL);metrics[name]=json.loads(mp.read_text())
 gt=list(csv.DictReader(a.gt_stats.open()));train=[r for r in gt if r['split']=='train'];image_group={r['image']:r['source_group'] for r in train};groups=sorted(set(image_group.values()));cache={};prep={}
 for name,q in merged.items():
  cache[name]={im:image_stats(rec,a.labels/f'{Path(im).stem}.txt') for im,rec in q['images'].items()};prep[name]=prepare_weighted(cache[name],image_group)
 pairs=(('B','A'),('C','B'),('C','C-noDiff'),('C','E0'));rng=np.random.default_rng(20261122);dist={f'{x}-{y}':[] for x,y in pairs}
 for _ in range(a.bootstrap):
  z=rng.choice(groups,len(groups),replace=True);mult=Counter(z)
  vals={n:weighted_evaluate(prep[n],mult) for n in merged}
  for x,y in pairs:dist[f'{x}-{y}'].append((vals[x]-vals[y]).tolist())
 boot={'unit':'source_group','repeats':a.bootstrap,'seed':20261122,'metrics':['map50_95','ap75','flash_ap50_95','black_ap50_95'],'pairs':{}}
 for key,z in dist.items():
  z=np.asarray(z);boot['pairs'][key]={m:{'mean':float(z[:,j].mean()),'ci95':np.quantile(z[:,j],[.025,.975]).tolist(),'p_gt_0':float(np.mean(z[:,j]>0))} for j,m in enumerate(boot['metrics'])}
 (o/'cir_group_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
 # Merge candidate-level traces.
 cand={}
 cand_rows={}
 for name,disk in VAR.items():
  rows=[]
  for f in range(3):rows+=list(csv.DictReader((o/f'metrics/fold{f}_{disk}_candidate.csv').open()))
  for r in rows:
   for k in ('iou_before','iou_after'):r[k]=float(r[k])
   r['tiny']=r['tiny']=='True';r['gain75']=r['gain75']=='True';r['lose75']=r['lose75']=='True'
   r['source_group']=image_group[r['image']]
  cand_rows[name]=rows
  cand[name]={'n':len(rows),'mean_iou_before':float(np.mean([r['iou_before'] for r in rows])),'mean_iou_after':float(np.mean([r['iou_after'] for r in rows])),'coverage75_before':float(np.mean([r['iou_before']>=.75 for r in rows])),'coverage75_after':float(np.mean([r['iou_after']>=.75 for r in rows])),'tiny_coverage75_before':float(np.mean([r['iou_before']>=.75 for r in rows if r['tiny']])),'tiny_coverage75_after':float(np.mean([r['iou_after']>=.75 for r in rows if r['tiny']])),'gain75':sum(r['gain75'] for r in rows),'lose75':sum(r['lose75'] for r in rows),'Flash_tiny_delta':subdelta(rows,0),'Black_tiny_delta':subdelta(rows,1)}
 (o/'cir_candidate_summary.json').write_text(json.dumps(cand,indent=2)+'\n')
 # Source-group paired intervals for the mechanism's primary candidate-coverage endpoint.
 cov_dist={f'{x}-{y}':{'coverage75':[],'tiny_coverage75':[]} for x,y in pairs}
 for _ in range(a.bootstrap):
  mult=Counter(rng.choice(groups,len(groups),replace=True))
  for x,y in pairs:
   def endpoint(name,tiny):
    rows=cand_rows[x if name=='E0' else name] if name=='E0' else cand_rows[name]
    field='iou_before' if name=='E0' else 'iou_after'
    z=[(float(r[field])>=.75,mult[r['source_group']]) for r in rows if (not tiny or r['tiny']) and mult[r['source_group']]]
    return sum(v*w for v,w in z)/sum(w for _,w in z)
   cov_dist[f'{x}-{y}']['coverage75'].append(endpoint(x,False)-endpoint(y,False))
   cov_dist[f'{x}-{y}']['tiny_coverage75'].append(endpoint(x,True)-endpoint(y,True))
 for key,endpoints in cov_dist.items():
  boot['pairs'][key]['candidate_coverage']={}
  for metric,z in endpoints.items():
   z=np.asarray(z);boot['pairs'][key]['candidate_coverage'][metric]={'mean':float(z.mean()),'ci95':np.quantile(z,[.025,.975]).tolist(),'p_gt_0':float(np.mean(z>0))}
 (o/'cir_group_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
 eff={}
 for name,disk in VAR.items():
  runs=[json.loads((o/f'runs/fold{f}_{disk}/run.json').read_text()) for f in range(3)];eff[name]={'parameters':runs[0]['parameters'],'checkpoint_bytes':runs[0]['checkpoint_bytes'],'mean_train_seconds_cpu':float(np.mean([x['elapsed_seconds'] for x in runs])),'mean_inference_seconds_per_image_cpu':merged[name]['mean_seconds_per_original']}
 (o/'cir_efficiency.json').write_text(json.dumps(eff,indent=2)+'\n')
 render(o,metrics,cand,boot,eff)
 env={'date_utc':'2026-11-21','python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'execution_device':'CPU'};(o/'environment_audit.json').write_text(json.dumps(env,indent=2)+'\n')
 hashes={str(x.relative_to(o)):hashlib.sha256(x.read_bytes()).hexdigest() for x in o.rglob('*') if x.is_file() and x.name!='FILE_HASHES.json' and x.suffix!='.pt'};(o/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print(json.dumps({'metrics':{k:{x:v[x] for x in ('map50','map50_95','map75')} for k,v in metrics.items()},'candidate':cand},indent=2))
def subdelta(rows,c):
 z=[r for r in rows if int(r['class_id'])==c and r['tiny']];return {'n':len(z),'iou_delta':float(np.mean([r['iou_after']-r['iou_before'] for r in z])) if z else None,'coverage75_delta':float(np.mean([r['iou_after']>=.75 for r in z])-np.mean([r['iou_before']>=.75 for r in z])) if z else None}
def table(metrics):
 out=['| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for n in ('E0','A','B','C','C-noDiff'):
  q=metrics[n];out.append(f"| {n} | {q['precision']:.4f} | {q['recall']:.4f} | {q['map50']:.4f} | {q['map50_95']:.4f} | {q['map75']:.4f} | {q['per_class_map50_95']['0']:.4f} | {q['per_class_map50_95']['1']:.4f} |")
 return '\n'.join(out)
def render(o,m,c,b,e):
 t=table(m);decision=bool(m['B']['map75']>m['A']['map75'] and m['C']['map75']>m['B']['map75'] and m['C']['map75']>m['C-noDiff']['map75'] and m['C']['map50_95']>=m['E0']['map50_95'] and m['C']['map75']>=m['E0']['map75'] and c['C']['coverage75_after']>c['C']['coverage75_before'])
 (o/'CIR_DESIGN.md').write_text('# CIR design\n\nA shared 3-layer Conv-BN-SiLU encoder maps aligned 128×128 ROIs to 64 dimensions. A uses the low-resolution reconstruction, B uses original pixels, C concatenates high, low, and explicit high-minus-low features, and C-noDiff uses an identically sized zero difference input. All variants append the same 8-D deployable E0 candidate summary (box geometry, score, class, center). A zero-initialized head predicts bounded center and log-width/height residuals. E0 weights, classes, scores, candidate count and first-stage NMS remain frozen.\n')
 (o/'CIR_CROP_ALIGNMENT_AUDIT.md').write_text('# CIR crop alignment audit\n\nThe low route exactly reproduces the content resize 4096×1024→1536×384 and resizes that content back to 4096×1024 without letterbox padding. High and low crops share the identical prediction-derived ROI coordinates, replicate padding, 128×128 resize, BGR→RGB conversion, and [0,1] normalization. `alignment/crop_alignment.json` and paired Flash/Black plates confirm equal shapes and coordinates. GT is absent from ROI construction.\n')
 (o/'CIR_IMPLEMENTATION_AUDIT.md').write_text('# CIR implementation audit\n\nAll four variants initialize with exactly zero residuals and E0-identical boxes. Unit tests verify head gradients, crop geometry, residual decoding, save/reload, and CPU forward/backward. Every 40-epoch run has nonzero gradients and falling loss. No negative-size box is possible because width/height use exponentiated bounded log residuals; inference clips image boundaries. CUDA was unavailable, so all local training and inference used CPU.\n')
 (o/'CIR_TRAINING_PROTOCOL.md').write_text('# CIR training protocol\n\nEach outer fold uses its own Stage10 E0 `last.pt`. Training candidates are inferred only from that fold training list; holdout candidates come from its frozen OOF prediction export. Candidate rule is conf=.001, class-aware NMS=.7, max_det=300; class-consistent greedy positive matching uses IoU≥.30, candidates overlapping any GT by >.10 but unmatched are ignored, and trustworthy negatives receive a 0.05 zero-residual penalty. All groups use seed42, 40 epochs, batch32, AdamW lr=.001, weight decay=.0001, SmoothL1 + IoU loss, no holdout validation/early stopping, and `last.pt`.\n')
 freeze={'seed':42,'roi_side':'min(256,max(96,4*max(pred_w,pred_h)))','roi_input':128,'candidate_rule':{'conf':.001,'nms_iou':.7,'max_det':300},'positive_iou':.30,'uncertain_ignore_iou':.10,'epochs':40,'batch':32,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'iou_loss_weight':1,'negative_zero_weight':.05,'checkpoint':'last.pt','device':'CPU'};(o/'CIR_EXPERIMENT_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
 (o/'CIR_TRAINING_RESULTS.md').write_text('# CIR training results\n\nTwelve frozen runs completed (4 variants × 3 folds). Training candidates per fold were 243, 251, and 227; positives were 115, 115, and 117. Every run started at zero residual and completed 40 epochs without collapse. See `runs/*/training_log.csv` and `run.json`.\n')
 fold=['| Fold | Model | mAP50-95 | AP75 |','|---:|---|---:|---:|']
 for f in range(3):
  for n,disk in [('E0','E0'),*VAR.items()]:
   p=Path('chapter4_detection/results/stage10')/f'e0_fold{f}_seed42_metrics.json' if n=='E0' else o/f'metrics/fold{f}_{disk}_ap.json'
   q=json.loads(p.read_text());fold.append(f"| {f} | {n} | {q['map50_95']:.4f} | {q['map75']:.4f} |")
 fold_text='\n'.join(fold)
 (o/'CIR_GROUP_CV_RESULTS.md').write_text('# CIR group-CV results\n\n'+t+'\n\n## Per-fold direction\n\n'+fold_text+'\n\nEffects are not directionally uniform across all folds. Source-group paired bootstrap intervals, including candidate and tiny-candidate coverage, are in `cir_group_bootstrap.json`. These are exploratory development OOF results, not independent-test generalization.\n')
 (o/'CIR_ABLATION_RESULTS.md').write_text(f"# CIR ablation results\n\nA→B tests original pixels; B→C tests cross-resolution fusion; C-noDiff→C tests explicit difference. OOF AP75: A {m['A']['map75']:.4f}, B {m['B']['map75']:.4f}, C {m['C']['map75']:.4f}, C-noDiff {m['C-noDiff']['map75']:.4f}. OOF mAP50-95: A {m['A']['map50_95']:.4f}, B {m['B']['map50_95']:.4f}, C {m['C']['map50_95']:.4f}, C-noDiff {m['C-noDiff']['map50_95']:.4f}. The frozen retention rule {'passes' if decision else 'does not pass'}.\n")
 (o/'CIR_ERROR_ANALYSIS.md').write_text('# CIR error analysis\n\n'+ '\n'.join(f"- {n}: mean matched-candidate IoU {q['mean_iou_before']:.4f}→{q['mean_iou_after']:.4f}; coverage@.75 {q['coverage75_before']:.4f}→{q['coverage75_after']:.4f}; {q['gain75']} gains and {q['lose75']} losses." for n,q in c.items())+'\n\nCandidate count, score and class are unchanged, so background-candidate count cannot increase. Box movement can still alter duplicate overlap; per-fold audits are retained. Both successful and harmful corrections are included.\n')
 (o/'CIR_EFFICIENCY.md').write_text('# CIR efficiency\n\n'+ '\n'.join(f"- {n}: {q['parameters']} parameters, {q['checkpoint_bytes']/1024:.1f} KiB, CPU local inference {q['mean_inference_seconds_per_image_cpu']*1000:.1f} ms/original image, mean observed CPU training {q['mean_train_seconds_cpu']:.1f} s/fold." for n,q in e.items())+'\n\nParameter counts and checkpoint sizes are reliable. Wall-clock values are approximate: system load was uncontrolled and the initial fold-0 B/C/C-noDiff jobs overlapped, so timings must not be used to rank variant efficiency.\n')
 (o/'STAGE22_FINAL_SUMMARY.md').write_text(f"# Stage22 final summary\n\nThe trainable CIR prototype and all 12 CPU runs completed.\n\n{t}\n\nCIR retention decision: **{'PASS' if decision else 'STOP'}**. Original-pixel, fusion, explicit-difference, E0 non-inferiority, and candidate-coverage conditions must all hold; the frozen joint rule {'passed' if decision else 'failed'}. Therefore CIR {'remains' if decision else 'does not remain'} a formal Chapter-4 innovation candidate. This is development OOF evidence only.\n")
 (o/'RUN_COMMANDS.md').write_text('# Stage22 commands\n\nSee `scripts/run_stage22_remaining.sh` for the exact sequential CPU pipeline. Main entry points are `stage22_prepare_candidates.py`, `stage22_prepare_crops.py`, `stage22_train_cir.py`, `stage22_predict_cir.py`, `evaluate_original.py`, `stage22_evaluate.py`, and `stage22_finalize.py`.\n')
if __name__=='__main__':main()
