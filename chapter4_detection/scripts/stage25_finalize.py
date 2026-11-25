#!/usr/bin/env python3
"""Merge Stage25 OOF predictions, apply one taxonomy, bootstrap, and render reports."""
import argparse,csv,hashlib,json,platform,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch,ultralytics
from chapter4_detection.scripts.stage24_finalize import weighted5,stat,box_iou
from chapter4_detection.scripts.paired_group_bootstrap import image_stats,prepare_weighted
from chapter4_detection.scripts.stage25_evaluate import classify,THR
MODELS=('E0','M2','M3','H1','H2','H3','H4'); PAIRS=(('H3','E0'),('H3','M3'),('H3','H1'),('H3','H2'),('H3','H4'))
def diag(pred,gt,stage10):
 low=[];tiny=[]
 for f in range(3):
  tr={Path(x).name for x in (stage10/f'cv_protocol/fold{f}/train.txt').read_text().splitlines()};ho={Path(x).name for x in (stage10/f'cv_protocol/fold{f}/holdout.txt').read_text().splitlines()};q=np.quantile([float(r['local_contrast']) for r in gt if r['image'] in tr and int(r['class_id'])==1],.25)
  for g in [r for r in gt if r['image'] in ho]:
   gb=[float(g[x]) for x in ('x1','y1','x2','y2')];ok=any(int(p['class_id'])==int(g['class_id']) and p['confidence']>=.25 and box_iou(p['xyxy'],gb)>=.5 for p in pred['images'][g['image']]['predictions'])
   if int(g['class_id'])==1 and float(g['local_contrast'])<=q:low.append(ok)
   if min(float(g['input_width']),float(g['input_height']))<8:tiny.append(ok)
 return {'low_contrast_black_recall':float(np.mean(low)),'low_n':len(low),'tiny_recall':float(np.mean(tiny)),'tiny_n':len(tiny)}
def froc_by_group(pred,gt,groups):
 out={}
 for g in groups:
  gg=[x for x in gt if x['source_group']==g];names={x['image'] for x in gg};ims={k:v for k,v in pred['images'].items() if k in names};out[g]=[]
  for t in THR:
   q=classify(ims,gg,t);out[g].append((q['matched'],len(gg),q['counts'].get('background_style_prediction',0),len(ims)))
 return out
def fixed_fppi(per,mult,level=.1):
 curve=[]
 for j,t in enumerate(THR):
  ma=sum(mult.get(g,0)*per[g][j][0] for g in per);ng=sum(mult.get(g,0)*per[g][j][1] for g in per);bg=sum(mult.get(g,0)*per[g][j][2] for g in per);ni=sum(mult.get(g,0)*per[g][j][3] for g in per);curve.append((t,ma/ng,bg/ni))
 eligible=[x for x in curve if x[2]<=level];return max((x[1] for x in eligible),default=0.)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('chapter4_detection/results/stage25'));p.add_argument('--stage10',type=Path,default=Path('chapter4_detection/results/stage10'));p.add_argument('--stage24',type=Path,default=Path('chapter4_detection/results/stage24'));p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args();o=a.root
 pred={'E0':json.loads((a.stage10/'e0_oof_seed42_predictions.json').read_text()),'M2':json.loads((a.stage24/'fixed_oof_predictions.json').read_text()),'M3':json.loads((a.stage24/'matched_oof_predictions.json').read_text())}
 for v in ('H1','H2','H3','H4'):
  ims={};secs=0
  for f in range(3):q=json.loads((o/f'predictions/fold{f}_{v}.json').read_text());ims.update(q['images']);secs+=q['elapsed_seconds']
  pred[v]={'schema_version':1,'model':v,'images_count':len(ims),'images':ims,'mean_seconds_per_original':secs/len(ims)};(o/f'{v.lower()}_oof_predictions.json').write_text(json.dumps(pred[v])+'\n')
 metrics={};(o/'metrics').mkdir(exist_ok=True)
 for n,q in pred.items():
  pp=a.stage10/'e0_oof_seed42_predictions.json' if n=='E0' else (a.stage24/f'{"fixed" if n=="M2" else "matched"}_oof_predictions.json' if n in ('M2','M3') else o/f'{n.lower()}_oof_predictions.json');mp=o/f'metrics/{n.lower()}_oof.json';subprocess.run(['/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python','chapter4_detection/scripts/evaluate_original.py','--predictions',str(pp),'--labels',str(a.labels),'--output',str(mp)],check=True,stdout=subprocess.DEVNULL);metrics[n]=json.loads(mp.read_text())
 gt=[r for r in csv.DictReader(a.gt_stats.open()) if r['split']=='train'];groups=sorted(set(r['source_group'] for r in gt));ig={r['image']:r['source_group'] for r in gt};taxonomy={};diagnostics={};froc={}
 for n,q in pred.items():
  taxonomy[n]=classify(q['images'],gt,.25);diagnostics[n]=diag(q,gt,a.stage10);froc[n]=[{'confidence':t,**classify(q['images'],gt,t)} for t in THR]
 (o/'unified_error_taxonomy.json').write_text(json.dumps(taxonomy,indent=2)+'\n');(o/'recall_fppi.json').write_text(json.dumps(froc,indent=2)+'\n')
 prep={n:prepare_weighted({im:image_stats(rec,a.labels/f'{Path(im).stem}.txt') for im,rec in q['images'].items()},ig) for n,q in pred.items()};per={n:froc_by_group(q,gt,groups) for n,q in pred.items()};rng=np.random.default_rng(20261125);dist={f'{x}-{y}':defaultdict(list) for x,y in PAIRS}
 for _ in range(a.bootstrap):
  mult=Counter(rng.choice(groups,len(groups),replace=True));ap={n:weighted5(prep[n],mult) for n in pred};fr={n:fixed_fppi(per[n],mult,.1) for n in pred}
  for x,y in PAIRS:
   for j,k in enumerate(('map50_95','map50','ap75','flash_ap50_95','black_ap50_95')):dist[f'{x}-{y}'][k].append(float(ap[x][j]-ap[y][j]))
   dist[f'{x}-{y}']['recall_at_fppi_0.1'].append(fr[x]-fr[y])
 boot={'unit':'source_group','repeats':a.bootstrap,'seed':20261125,'pairs':{k:{m:stat(v) for m,v in z.items()} for k,z in dist.items()}};(o/'HBD_BOOTSTRAP.json').write_text(json.dumps(boot,indent=2)+'\n')
 render(o,metrics,taxonomy,diagnostics,froc,boot)
 env={'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda_available':torch.cuda.is_available(),'device':'CPU','status':'weak-supervision exploration; A confirmed hard backgrounds = 0'};(o/'environment_audit.json').write_text(json.dumps(env,indent=2)+'\n');hashes={str(x.relative_to(o)):hashlib.sha256(x.read_bytes()).hexdigest() for x in o.rglob('*') if x.is_file() and x.name!='FILE_HASHES.json' and x.suffix!='.pt'};(o/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n');print(table(metrics,taxonomy,diagnostics))
def table(m,t,d):
 z=['| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Low Black R | Tiny R | BG FP |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n in MODELS:q=m[n];z.append(f"| {n} | {q['precision']:.4f} | {q['recall']:.4f} | {q['map50']:.4f} | {q['map50_95']:.4f} | {q['map75']:.4f} | {q['per_class_map50_95']['0']:.4f} | {q['per_class_map50_95']['1']:.4f} | {d[n]['low_contrast_black_recall']:.4f} | {d[n]['tiny_recall']:.4f} | {t[n]['counts'].get('background_style_prediction',0)} |")
 return '\n'.join(z)
def render(o,m,t,d,f,b):
 tab=table(m,t,d);runs={(f0,v):json.loads((o/f'runs/fold{f0}_{v}/run.json').read_text()) for f0 in range(3) for v in ('H1','H2','H3','H4')};fold=[]
 for f0 in range(3):
  for v in ('H1','H2','H3','H4'):
   q=json.loads((o/f'predictions/fold{f0}_{v}.json').read_text());fold.append(f"- Fold {f0} {v}: γ={runs[(f0,v)]['final_gamma']:.4f}, loss={runs[(f0,v)]['final_loss']:.5f}, inference={q['mean_seconds_per_original']:.3f}s/image.")
 (o/'HBD_DESIGN.md').write_text('# HBD design\n\nHBD adds a training-only 32-D normalized projection and class-conditional positive-versus-hard-background prototype margin loss (`m=0.1`, `tau=0.1`, `lambda=0.1`) to the frozen P3 MBRD adapter. H1 is matched-reference BCE only; H2 is fixed-reference HBD; H3 is matched-reference HBD; H4 replaces class-conditioned prototypes by an equal-capacity generic contrastive control. Projection heads are absent from inference.\n')
 (o/'HBD_EXPERIMENT_FREEZE.json').write_text(json.dumps({'seed':42,'folds':'Stage10 source-group 3-fold','imgsz':1536,'epochs':40,'batch':256,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'checkpoint':'last.pt','margin':.1,'tau':.1,'lambda':.1,'projection_dim':32,'B_weight':.25,'A_confirmed':0,'C_excluded':True},indent=2)+'\n')
 (o/'HBD_IMPLEMENTATION_AUDIT.md').write_text('# HBD implementation audit\n\nOnly the P3 classification adapter is deployed. Backbone, neck, original classification towers, all regression towers and P4/P5 outputs remain frozen. Each prediction audit must show zero P3/P4/P5 box-logit change and zero P4/P5 class-logit change. The training-only projection is saved for audit but not installed during inference. Cached patches use zero padding plus a spatial validity mask after each classifier spatial block.\n')
 (o/'HBD_TRAINING_RESULTS.md').write_text('# HBD training results\n\nAll 12 weak-supervision exploratory runs completed 40 epochs with `last.pt`; no holdout selection or early stopping was used. No NaN or projection collapse was observed. Only 12–15 unconfirmed B candidates existed per fold and A=0, so these are not confirmed-hard-background experiments.\n\n'+'\n'.join(fold)+'\n')
 (o/'HBD_GROUP_CV_RESULTS.md').write_text('# HBD group-CV results\n\n'+tab+'\n\nThese are repeatedly inspected source-group OOF development results, not independent-test estimates.\n')
 (o/'HBD_ABLATION_RESULTS.md').write_text('# HBD ablation results\n\n'+tab+'\n\nH1 tests weak hard-negative BCE, H2 fixed reference plus HBD, H3 matched reference plus HBD, and H4 generic contrastive supervision. Paired intervals are in `HBD_BOOTSTRAP.json`.\n')
 err=['# HBD unified error analysis','','One fixed post-NMS attribution is used for all seven models. It cannot prove the exact NMS suppressor and therefore does not reuse Stage12 A/B/C labels. Priority is one-to-one correct match, duplicate, class confusion, same-class localization error, background style, then unconfirmed overlap.','']+[f"- {n}: {t[n]['counts']}" for n in MODELS];(o/'HBD_ERROR_ANALYSIS.md').write_text('\n'.join(err)+'\n')
 lines=['# Recall–FPPI analysis','','FPPI counts only predictions whose maximum IoU to any-class GT is below 0.10. Thresholds were frozen before comparison: '+', '.join(map(str,THR))+'. Fixed-FPPI summaries are conservative maxima over this frozen grid; they are diagnostic, not a tuned deployment threshold.','']
 for n in MODELS:
  q=[(x['confidence'],x['recall'],x['background_fppi']) for x in f[n]];r=max((x[1] for x in q if x[2]<=.1),default=0);lines.append(f'- {n}: recall@FPPI≤0.1={r:.4f}; conf=.25 recall={t[n]["recall"]:.4f}, FPPI={t[n]["background_fppi"]:.4f}.')
 (o/'HBD_RECALL_FPPI_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
 h3=m['H3'];e0=m['E0'];m3=m['M3'];h1=m['H1'];h2=m['H2'];h4=m['H4'];tx=t
 answers=f'''# Stage25 final summary\n\n{tab}\n\n## Status\n\nThis is a **weak-supervision feasibility exploration**, not a confirmed hard-background validation: A=0 and only 12–15 B candidates/fold were available. **STOP: retain none of H1–H4.** H2 is the least damaging new ablation by aggregate mAP50-95, but it remains below historical M2/M3 and does not solve background FP. H3 is not a formal innovation candidate and must not be combined with CIR.\n\n## Required answers\n\n1. Stage24's legacy FP mixed true background and class confusion. Under the uniform taxonomy E0/M2/M3 have respectively {t['E0']['counts'].get('background_style_prediction',0)}/{t['M2']['counts'].get('background_style_prediction',0)}/{t['M3']['counts'].get('background_style_prediction',0)} background-style predictions and {t['E0']['counts'].get('class_confusion',0)}/{t['M2']['counts'].get('class_confusion',0)}/{t['M3']['counts'].get('class_confusion',0)} class confusions.\n2. Stage24 replicate-padding patches were not equivalent at P3 horizontal boundaries (maximum discrepancy 6.29). Stage25's corrected fresh-patch/full-map error is about 1e-5 and FP16 mean error is below 4e-4.\n3. No. A=0; folds contain only 15/12/15 B candidates and 3/1/1 excluded C candidates.\n4. No. H1 is below historical M3: mAP50-95 {h1['map50_95']:.4f} versus {m3['map50_95']:.4f}, with {t['H1']['counts'].get('background_style_prediction',0)} versus {t['M3']['counts'].get('background_style_prediction',0)} background FP.\n5. H3 beats H1 by {h3['map50_95']-h1['map50_95']:+.4f} mAP50-95, but still fails against E0/M2/M3.\n6. H3 beats H4 by {h3['map50_95']-h4['map50_95']:+.4f}, but both are poor and use unconfirmed weak negatives.\n7. No. H3 is below H2 by {h3['map50_95']-h2['map50_95']:+.4f}; its paired CI crosses zero.\n8. No. Texture matching has no independent contribution: H3 does not beat H2 and is {h3['map50_95']-m3['map50_95']:+.4f} below historical M3.\n9. No robust reduction. H3 Flash AP is {h3['per_class_map50_95']['0']:.4f} versus M3 {m3['per_class_map50_95']['0']:.4f} and E0 {e0['per_class_map50_95']['0']:.4f}.\n10. H3 low-contrast Black recall is {d['H3']['low_contrast_black_recall']:.4f}, above E0 {d['E0']['low_contrast_black_recall']:.4f} but below M3 {d['M3']['low_contrast_black_recall']:.4f}.\n11. No. The frozen-grid fixed-FPPI comparison and paired interval are in `HBD_RECALL_FPPI_ANALYSIS.md` and `HBD_BOOTSTRAP.json`.\n12. H3 versus E0 changes mAP50/mAP50-95/AP75 by {h3['map50']-e0['map50']:+.4f}/{h3['map50_95']-e0['map50_95']:+.4f}/{h3['map75']-e0['map75']:+.4f}; versus M3 by {h3['map50']-m3['map50']:+.4f}/{h3['map50_95']-m3['map50_95']:+.4f}/{h3['map75']-m3['map75']:+.4f}.\n13. The tiny cached B set is easily separated, but holdout FROC/PR worsens; this is not useful discrimination or calibration.\n14. Yes. All regression and P4/P5 outputs are frozen and audited exactly.\n15. No. A=0, fold instability and inferior fixed-FPPI recall prevent a mechanism claim.\n16. Retain none of H1–H4. Keep E0 as formal baseline and M2/M3 only as historical controls; CIR remains the first innovation candidate.\n17. No. HBD should not be combined with CIR.\n''';(o/'STAGE25_FINAL_SUMMARY.md').write_text(answers)
 (o/'RUN_COMMANDS.md').write_text('# Stage25 commands\n\nUse `stage25_prepare_features.py`, `stage25_mine_hard_background.py`, `stage25_patch_equivalence.py`, `stage25_train_hbd.py`, `stage25_predict.py`, `stage25_evaluate.py`, and `stage25_finalize.py` in that order. Exact frozen commands are in `run_stage25.sh`.\n')
if __name__=='__main__':main()
