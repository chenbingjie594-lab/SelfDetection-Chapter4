#!/usr/bin/env python3
"""Merge Stage23 OOF predictions, evaluate, bootstrap, audit gates and render reports."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
from chapter4_detection.scripts.paired_group_bootstrap import image_stats,prepare_weighted,weighted_evaluate
from chapter4_detection.scripts.stage23_common import load_gt,best_rows,summarize,iou

def merge_fold(files,name):
 images={};secs=0
 for p in files:q=json.loads(p.read_text());images.update(q['images']);secs+=q.get('elapsed_seconds',0)
 return {'schema_version':1,'model':name,'protocol':'three-fold source-group OOF; geometry-only gate','images_count':len(images),'images':images,'elapsed_seconds':secs}
def cov_boot(rows,group_draws):
 by=defaultdict(list)
 for r in rows:by[r['source_group']].append(r)
 z=[]
 for mult in group_draws:
  sample=[r for g,n in mult.items() for r in by[g] for _ in range(n)];tiny=[r for r in sample if r['tiny']];z.append([np.mean([r['best_iou_after']>=.75 for r in sample]),np.mean([r['best_iou_after']>=.75 for r in tiny])])
 return np.asarray(z)
def stat(z):return {'mean':float(z.mean()),'ci95':np.quantile(z,[.025,.975]).tolist(),'p_gt_0':float(np.mean(z>0))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('chapter4_detection/results/stage23'));p.add_argument('--stage10',type=Path,default=Path('chapter4_detection/results/stage10'));p.add_argument('--stage22',type=Path,default=Path('chapter4_detection/results/stage22'));p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args();o=a.root
 pred={'E0':json.loads((a.stage10/'e0_oof_seed42_predictions.json').read_text()),'C':json.loads((a.stage22/'C_oof_predictions.json').read_text())}
 for n,d in [('C-Half','half'),('C-GateGeo','geo'),('C-GateFull','full')]:pred[n]=merge_fold([o/f'predictions/fold{f}_{d}.json' for f in range(3)],n);(o/f'{d}_oof_predictions.json').write_text(json.dumps(pred[n])+'\n')
 # Oracle-Gate chooses E0 or C per candidate using GT only as a diagnostic upper bound.
 gt=[r for r in load_gt(a.gt_stats) if r['split']=='train'];gtim=defaultdict(list)
 for g in gt:gtim[g['image']].append(g)
 oracle=json.loads(json.dumps(pred['E0']))
 for im,rec in oracle['images'].items():
  cp=pred['C']['images'][im]['predictions']
  for k,bp in enumerate(rec['predictions']):
   gs=[g for g in gtim[im] if int(g['class_id'])==int(bp['class_id'])];gb=[[float(g[x]) for x in ('x1','y1','x2','y2')] for g in gs];q0=max([iou(bp['xyxy'],x) for x in gb],default=0);q1=max([iou(cp[k]['xyxy'],x) for x in gb],default=0)
   if q1>q0:bp['xyxy']=cp[k]['xyxy']
 oracle['model']='Oracle-Gate (diagnostic only; GT-dependent)';pred['Oracle-Gate']=oracle;(o/'oracle_gate_oof_predictions.json').write_text(json.dumps(oracle)+'\n')
 metrics={};rows={}
 for n,q in pred.items():
  disk=n.lower().replace('-','_');pp=o/f'{disk}_tmp.json';pp.write_text(json.dumps(q)+'\n');mp=o/f'{disk}_oof_metrics.json';subprocess.run(['/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python','chapter4_detection/scripts/evaluate_original.py','--predictions',str(pp),'--labels',str(a.labels),'--output',str(mp)],check=True,stdout=subprocess.DEVNULL);pp.unlink();metrics[n]=json.loads(mp.read_text());rows[n]=best_rows(pred['E0']['images'],q['images'],gt)
 # Retain fold-level AP and corrected localization for direction checks.
 (o/'metrics').mkdir(exist_ok=True);fold_metrics={}
 for f in range(3):
  allowed={Path(x).name for x in (a.stage10/f'cv_protocol/fold{f}/holdout.txt').read_text().splitlines()};fg=[g for g in gt if g['image'] in allowed];bp=a.stage10/f'e0_fold{f}_seed42_predictions.json'
  sources={'E0':bp,'C':a.stage22/f'predictions/fold{f}_C.json','C-Half':o/f'predictions/fold{f}_half.json','C-GateGeo':o/f'predictions/fold{f}_geo.json','C-GateFull':o/f'predictions/fold{f}_full.json'}
  for n,pp in sources.items():
   mp=o/f"metrics/fold{f}_{n.lower().replace('-','_')}_ap.json";subprocess.run(['/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python','chapter4_detection/scripts/evaluate_original.py','--predictions',str(pp),'--labels',str(a.labels),'--output',str(mp)],check=True,stdout=subprocess.DEVNULL);fold_metrics[(f,n)]=json.loads(mp.read_text());rr=best_rows(bp,pp,fg);(o/f"metrics/fold{f}_{n.lower().replace('-','_')}_candidate.json").write_text(json.dumps({'summary':summarize(rr),'rows':rr},indent=2)+'\n')
 corrected=json.loads((o/'STAGE22_CORRECTED_CANDIDATE_METRICS.json').read_text());stage22_names=['E0','A','B','C','C-noDiff'];all22={n:corrected[n]['rows'] for n in stage22_names}
 scene={n:scene_audit(q['images'],gt) for n,q in pred.items()};(o/'prediction_audit.json').write_text(json.dumps(scene,indent=2)+'\n')
 base_images=pred['E0']['images'];invariance={}
 for n,q in pred.items():
  same_count=all(len(q['images'][im]['predictions'])==len(rec['predictions']) for im,rec in base_images.items());same_cls=all([p['class_id'] for p in q['images'][im]['predictions']]==[p['class_id'] for p in rec['predictions']] for im,rec in base_images.items());same_score=all([p['confidence'] for p in q['images'][im]['predictions']]==[p['confidence'] for p in rec['predictions']] for im,rec in base_images.items());invariance[n]={'same_candidate_count':same_count,'same_class_order':same_cls,'same_scores':same_score}
 assert all(all(v.values()) for v in invariance.values());(o/'invariance_audit.json').write_text(json.dumps(invariance,indent=2)+'\n')
 image_group={r['image']:r['source_group'] for r in gt};groups=sorted(set(image_group.values()));cache={n:{im:image_stats(rec,a.labels/f'{Path(im).stem}.txt') for im,rec in q['images'].items()} for n,q in pred.items()};prep={n:prepare_weighted(x,image_group) for n,x in cache.items()};rng=np.random.default_rng(20261123)
 pairs=[('C-GateFull','C'),('C-GateFull','C-Half'),('C-GateFull','C-GateGeo'),('C-GateFull','E0')];apdist={f'{x}-{y}':[] for x,y in pairs}
 for _ in range(a.bootstrap):
  mult=Counter(rng.choice(groups,len(groups),replace=True));v={n:weighted_evaluate(prep[n],mult) for n in pred}
  for x,y in pairs:apdist[f'{x}-{y}'].append((v[x]-v[y]).tolist())
 group_draws=[Counter(rng.choice(groups,len(groups),replace=True)) for _ in range(a.bootstrap)];cov={n:cov_boot(r,group_draws) for n,r in rows.items()};boot={'unit':'source_group','repeats':a.bootstrap,'seed':20261123,'true_gt':len(gt),'pairs':{}}
 for x,y in pairs:
  ad=np.asarray(apdist[f'{x}-{y}']);cd=cov[x]-cov[y];boot['pairs'][f'{x}-{y}']={'map50_95':stat(ad[:,0]),'ap75':stat(ad[:,1]),'flash_ap50_95':stat(ad[:,2]),'black_ap50_95':stat(ad[:,3]),'coverage75':stat(cd[:,0]),'tiny_coverage75':stat(cd[:,1])}
 (o/'QPR_CIR_BOOTSTRAP.json').write_text(json.dumps(boot,indent=2)+'\n')
 # Corrected Stage22 coverage bootstrap plus the unchanged formal AP bootstrap.
 pairs22=[('C','E0'),('B','A'),('C','B'),('C','C-noDiff')];c22={n:cov_boot(r,group_draws) for n,r in all22.items()};old=json.loads((a.stage22/'cir_group_bootstrap.json').read_text());b22={'unit':'source_group','repeats':a.bootstrap,'seed':20261123,'pairs':{}}
 for x,y in pairs22:
  ap={k:v for k,v in old['pairs'][f'{x}-{y}'].items() if k in ('map50_95','ap75','flash_ap50_95','black_ap50_95')};b22['pairs'][f'{x}-{y}']={'formal_ap_metrics':ap,'corrected_coverage75':stat((c22[x]-c22[y])[:,0]),'corrected_tiny_coverage75':stat((c22[x]-c22[y])[:,1])}
 (o/'STAGE22_CORRECTED_BOOTSTRAP.json').write_text(json.dumps(b22,indent=2)+'\n')
 gate_analysis=analyse_gates(o,a.stage22,gt);(o/'gate_analysis.json').write_text(json.dumps(gate_analysis,indent=2)+'\n');render(o,metrics,rows,gate_analysis,boot,fold_metrics,scene)
 env={'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),'device':'CPU'};(o/'environment_audit.json').write_text(json.dumps(env,indent=2)+'\n');hashes={str(x.relative_to(o)):hashlib.sha256(x.read_bytes()).hexdigest() for x in o.rglob('*') if x.is_file() and x.name!='FILE_HASHES.json' and x.suffix!='.pt'};(o/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print(json.dumps({n:{k:metrics[n][k] for k in ('map50','map50_95','map75')} for n in metrics},indent=2))
def analyse_gates(o,s22,gt):
 gtim=defaultdict(list)
 for g in gt:gtim[g['image']].append(g)
 out={}
 for mode in ('geo','full'):
  data=[]
  for f in range(3):
   tr=json.loads((o/f'predictions/fold{f}_{mode}_trace.json').read_text());ct={x['candidate_id']:x for x in json.loads((s22/f'predictions/fold{f}_C_trace.json').read_text())}
   for x in tr:
    gs=[g for g in gtim[x['image']] if int(g['class_id'])==int(x['class_id'])];gb=[[float(g[z]) for z in ('x1','y1','x2','y2')] for g in gs];q0=max([iou(x['original_xyxy'],b) for b in gb],default=0);q1=max([iou(ct[x['candidate_id']]['refined_xyxy'],b) for b in gb],default=0)
    if q0>=.30:data.append({'gate':x['gate'],'delta_cir_iou':q1-q0,'class_id':x['class_id'],'original_iou':q0})
  useful=[x['gate'] for x in data if x['delta_cir_iou']>.01];harm=[x['gate'] for x in data if x['delta_cir_iou']<-.02];out[mode]={'n':len(data),'gate_mean':float(np.mean([x['gate'] for x in data])),'gate_std':float(np.std([x['gate'] for x in data])),'useful_n':len(useful),'useful_gate_mean':float(np.mean(useful)) if useful else None,'harmful_n':len(harm),'harmful_gate_mean':float(np.mean(harm)) if harm else None,'high_quality_gate_mean':float(np.mean([x['gate'] for x in data if x['original_iou']>=.75])),'Flash_gate_mean':float(np.mean([x['gate'] for x in data if x['class_id']==0])),'Black_gate_mean':float(np.mean([x['gate'] for x in data if x['class_id']==1]))}
 return out
def scene_audit(images,gt):
 gtim=defaultdict(list)
 for g in gt:gtim[g['image']].append(g)
 bg=dup=0
 for im,rec in images.items():
  gs=gtim[im];ps=rec['predictions']
  for p in ps:
   same=[g for g in gs if int(g['class_id'])==int(p['class_id'])];bg+=max([iou(p['xyxy'],[float(g[x]) for x in ('x1','y1','x2','y2')]) for g in same],default=0)<.10
  for g in gs:
   gb=[float(g[x]) for x in ('x1','y1','x2','y2')];dup+=sum(int(p['class_id'])==int(g['class_id']) and iou(p['xyxy'],gb)>=.5 for p in ps)>1
 return {'background_style_candidates_iou_lt_0.10':bg,'gt_with_duplicate_iou_ge_0.50':dup,'total_candidates':sum(len(x['predictions']) for x in images.values())}
def table(m,rows):
 z=['| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Coverage75 | Tiny cov. |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n in ('E0','C','C-Half','C-GateGeo','C-GateFull','Oracle-Gate'):
  q=m[n];s=summarize(rows[n]);z.append(f"| {n} | {q['precision']:.4f} | {q['recall']:.4f} | {q['map50']:.4f} | {q['map50_95']:.4f} | {q['map75']:.4f} | {q['per_class_map50_95']['0']:.4f} | {q['per_class_map50_95']['1']:.4f} | {s['coverage75_after_count']}/{s['gt']} | {s['tiny_coverage75_after_count']}/{s['tiny_gt']} |")
 return '\n'.join(z)
def render(o,m,rows,g,b,fold_metrics,scene):
 t=table(m,rows);cross=[json.loads((o/f'crossfit/fold{f}/crossfit_report.json').read_text()) for f in range(3)];runs={(f,x):json.loads((o/f'runs/fold{f}_{x}/run.json').read_text()) for f in range(3) for x in ('geo','full')}
 (o/'QPR_CIR_DESIGN.md').write_text('# QPR-CIR design\n\nA frozen Stage22 CIR-C residual is multiplied by a learned scalar gate. Gate-Geo uses 13 deployable geometry/residual features. Gate-Full uses the same 13 plus shared CIR high, low and absolute-difference feature summaries (205 inputs total). Both use a 32-unit SiLU MLP and sigmoid output initialized to g=0.5. GT-derived quality is training-only.\n')
 (o/'QPR_CIR_CROSSFIT_PROTOCOL.md').write_text('# QPR-CIR cross-fit protocol\n\nEach Stage10 outer training split is partitioned into three inner source-group folds. Nine temporary CIR-C models use the frozen Stage22 40-epoch recipe; each creates quality labels only for its unseen inner fold. The merged labels train the outer gate. Outer holdout labels are never loaded by gate training. The outer E0 detector did see outer-training images, so this is not fully independent detector evidence.\n')
 (o/'QPR_CIR_QUALITY_LABEL_AUDIT.md').write_text('# QPR-CIR quality-label audit\n\n'+ '\n'.join(f"- Outer fold {q['outer_fold']}: {q['positive']} matched, {q['background']} trustworthy background, {q['useful']} useful (ΔIoU>0.01), {q['harmful']} harmful (ΔIoU<-0.02), {q['neutral']} near-neutral; Flash {q['Flash']}, Black {q['Black']}, tiny {q['tiny']}." for q in cross)+'\n')
 (o/'QPR_CIR_IMPLEMENTATION_AUDIT.md').write_text('# QPR-CIR implementation audit\n\nUnit tests verify corrected best-candidate switching, preservation by an alternate candidate, g=0/0.5/1 endpoint semantics and gate gradients. Classes, scores, order and candidate count remain frozen. Width and height remain positive through CIR exponential decoding. All runs used CPU.\n')
 (o/'QPR_CIR_TRAINING_RESULTS.md').write_text('# QPR-CIR training results\n\nSix gates completed 40 epochs with last-checkpoint selection. Gate-Geo has '+str(runs[(0,'geo')]['parameters'])+' parameters; Gate-Full has '+str(runs[(0,'full')]['parameters'])+' parameters. Logs, SHA256 and checkpoints are retained under `runs/`.\n')
 ft=['| Fold | Model | mAP50-95 | AP75 |','|---:|---|---:|---:|']
 for f in range(3):
  for n in ('E0','C','C-Half','C-GateGeo','C-GateFull'):
   q=fold_metrics[(f,n)];ft.append(f"| {f} | {n} | {q['map50_95']:.4f} | {q['map75']:.4f} |")
 (o/'QPR_CIR_GROUP_CV_RESULTS.md').write_text('# QPR-CIR group-CV results\n\n'+t+'\n\n## Per-fold direction\n\n'+'\n'.join(ft)+'\n\nOracle-Gate is GT-dependent and diagnostic only. These are development OOF results, not independent-test estimates.\n')
 (o/'QPR_CIR_GATE_ANALYSIS.md').write_text('# QPR-CIR gate analysis\n\n'+ '\n'.join(f"- {n}: mean g={q['gate_mean']:.4f}±{q['gate_std']:.4f}; useful mean={q['useful_gate_mean']:.4f} (n={q['useful_n']}); harmful mean={q['harmful_gate_mean']:.4f} (n={q['harmful_n']}); high-quality mean={q['high_quality_gate_mean']:.4f}." for n,q in g.items())+'\n\n## Protection and scene audit\n\n'+ '\n'.join(f"- {n}: coverage gains={summarize(rows[n])['gain75']}, losses={summarize(rows[n])['lose75']}, background-style candidates={scene[n]['background_style_candidates_iou_lt_0.10']}, duplicate GT={scene[n]['gt_with_duplicate_iou_ge_0.50']}." for n in ('E0','C','C-Half','C-GateGeo','C-GateFull'))+'\n')
 (o/'QPR_CIR_ABLATION.md').write_text('# QPR-CIR ablation\n\n'+t+'\n\nC-Half isolates uniform attenuation; GateGeo tests deployable geometry-only selection; GateFull tests whether cross-resolution summaries add quality-prediction value. Paired intervals are in `QPR_CIR_BOOTSTRAP.json`.\n')
 full=summarize(rows['C-GateFull']);c=summarize(rows['C']);passed=m['C-GateFull']['map50_95']>=m['C']['map50_95'] and m['C-GateFull']['map75']>m['C']['map75'] and full['coverage75_after_count']>=c['coverage75_after_count'] and m['C-GateFull']['map50_95']>m['C-Half']['map50_95'] and m['C-GateFull']['map50_95']>m['C-GateGeo']['map50_95']
 (o/'STAGE23_FINAL_SUMMARY.md').write_text('# Stage23 final summary\n\n'+t+f"\n\nFrozen QPR-CIR decision: **{'PASS' if passed else 'STOP'}**. This joint point-estimate rule does not imply statistical significance; source-group intervals and fold direction must be considered. Stage22 remains STOP regardless of this correction.\n\n## Required conclusions\n\n1. The Stage22 coverage evaluator is corrected: it now independently re-selects the best same-class candidate after refinement.\n2. Corrected CIR-C coverage is 82/174 versus E0 80/174: 15 GT gain and 13 GT lose coverage, a net +2.\n3. Harmful corrections concentrate among already adequate/high-quality boxes; C loses 13 high-quality GT despite its 15 gains.\n4. Inference-available features show only modest quality separation. GateFull mean g is {g['full']['useful_gate_mean']:.4f} for useful versus {g['full']['harmful_gate_mean']:.4f} for harmful candidates.\n5. GateFull is not better than GateGeo: mAP50-95 {m['C-GateFull']['map50_95']:.4f} versus {m['C-GateGeo']['map50_95']:.4f}, AP75 {m['C-GateFull']['map75']:.4f} versus {m['C-GateGeo']['map75']:.4f}.\n6. GateFull is not better than C-Half: mAP50-95 {m['C-GateFull']['map50_95']:.4f} versus {m['C-Half']['map50_95']:.4f}, AP75 {m['C-GateFull']['map75']:.4f} versus {m['C-Half']['map75']:.4f}.\n7. GateFull reduces high-quality losses from 13 to {full['lose75']}, but also reduces gains from 15 to {full['gain75']}; coverage remains {full['coverage75_after_count']}/174.\n8. GateFull AP75 falls below C ({m['C-GateFull']['map75']:.4f} versus {m['C']['map75']:.4f}); mAP50-95 is also slightly lower ({m['C-GateFull']['map50_95']:.4f} versus {m['C']['map50_95']:.4f}).\n9. Flash AP falls relative to C ({m['C-GateFull']['per_class_map50_95']['0']:.4f} versus {m['C']['per_class_map50_95']['0']:.4f}), while Black AP improves ({m['C-GateFull']['per_class_map50_95']['1']:.4f} versus {m['C']['per_class_map50_95']['1']:.4f}); both classes do not jointly benefit.\n10. Tiny coverage degrades from C 17/52 to GateFull {full['tiny_coverage75_after_count']}/52.\n11. Background-style count stays {scene['C-GateFull']['background_style_candidates_iou_lt_0.10']}; duplicate-GT count is {scene['C-GateFull']['gt_with_duplicate_iou_ge_0.50']} versus C {scene['C']['gt_with_duplicate_iou_ge_0.50']}.\n12. The diagnostic Oracle-Gate shows headroom (mAP50-95 {m['Oracle-Gate']['map50_95']:.4f}, AP75 {m['Oracle-Gate']['map75']:.4f}), but it uses GT and is not a deployable model.\n13. Cross-resolution quality features do not establish an advantage over geometry-only gating. A generic sigmoid quality gate is not itself novel.\n14. QPR-CIR should stop as a formal Chapter-4 innovation candidate under the frozen Stage23 criteria.\n")
 (o/'RUN_COMMANDS.md').write_text('# Stage23 commands\n\nThe exact sequential pipeline is `bash chapter4_detection/scripts/run_stage23.sh`, followed by `stage23_finalize.py`. All training uses seed42, 40 epochs, batch32 and last checkpoints on CPU.\n')
 (o/'QPR_CIR_LITERATURE_POSITIONING.md').write_text('# QPR-CIR literature positioning\n\nA sigmoid gate, learned box refinement and localization-quality estimation are established ideas; QPR-CIR cannot claim novelty from gating alone. [IoU-Net (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Borui_Jiang_Acquisition_of_Localization_ECCV_2018_paper.html) predicts localization IoU for ranking/NMS and optimization-based refinement. [BBRefinement (2022)](https://doi.org/10.3390/app12073402) separately refines detector boxes from object crops. [CLQ (2024)](https://doi.org/10.1016/j.neucom.2024.127504) estimates localization quality to improve detection confidence. QPR-CIR instead freezes scores, classes, candidates and CIR, and predicts only how much of an existing residual to apply; its narrower hypothesis is that paired original/reconstructed CIR features reveal whether that residual is safe. GateFull does not outperform GateGeo or uniform attenuation here, so the cross-resolution quality-control distinction is not empirically established and novelty remains unconfirmed.\n')
if __name__=='__main__':main()
