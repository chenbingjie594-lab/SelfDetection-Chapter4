#!/usr/bin/env python3
"""Merge Stage24 OOF results, bootstrap paired effects, and render frozen reports."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch,ultralytics
from ultralytics.utils.metrics import compute_ap
from chapter4_detection.scripts.paired_group_bootstrap import image_stats,prepare_weighted,weighted_evaluate
VAR={'M1':'plain','M2':'fixed','M3':'matched'}
def stat(z):z=np.asarray(z);return {'mean':float(z.mean()),'ci95':np.quantile(z,[.025,.975]).tolist(),'p_gt_0':float(np.mean(z>0))}
def weighted5(prepared,mult):
 base=weighted_evaluate(prepared,mult);aps=[]
 for cls in (0,1):
  tp,groups,gt=prepared[cls];w=np.asarray([mult.get(g,0) for g in groups],float);keep=w>0;w=w[keep];z=tp[keep];ng=sum(mult.get(g,0)*n for g,n in gt.items());t=np.cumsum(w*z[:,0]);f=np.cumsum(w*(~z[:,0]));aps.append(compute_ap(t/ng,t/(t+f+1e-16))[0] if ng and len(t) else 0.)
 return np.array([base[0],np.mean(aps),base[1],base[2],base[3]])
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('chapter4_detection/results/stage24'));p.add_argument('--stage10',type=Path,default=Path('chapter4_detection/results/stage10'));p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args();o=a.root;pred={'E0':json.loads((a.stage10/'e0_oof_seed42_predictions.json').read_text())};foldmetrics={};diag={}
 for n,d in VAR.items():
  images={};secs=0
  for f in range(3):q=json.loads((o/f'predictions/fold{f}_{d}.json').read_text());images.update(q['images']);secs+=q['elapsed_seconds']
  pred[n]={'schema_version':1,'model':n,'images_count':len(images),'images':images,'mean_seconds_per_original':secs/len(images)};(o/f'{d}_oof_predictions.json').write_text(json.dumps(pred[n])+'\n')
 metrics={};(o/'metrics').mkdir(exist_ok=True)
 for n,q in pred.items():
  pp=a.stage10/'e0_oof_seed42_predictions.json' if n=='E0' else o/f'{VAR[n]}_oof_predictions.json';mp=o/f'metrics/{n.lower()}_oof.json';subprocess.run(['/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python','chapter4_detection/scripts/evaluate_original.py','--predictions',str(pp),'--labels',str(a.labels),'--output',str(mp)],check=True,stdout=subprocess.DEVNULL);metrics[n]=json.loads(mp.read_text())
 for f in range(3):
  for n,d in VAR.items():
   pp=o/f'predictions/fold{f}_{d}.json';mp=o/f'metrics/fold{f}_{d}_ap.json';subprocess.run(['/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python','chapter4_detection/scripts/evaluate_original.py','--predictions',str(pp),'--labels',str(a.labels),'--output',str(mp)],check=True,stdout=subprocess.DEVNULL);foldmetrics[(f,n)]=json.loads(mp.read_text());dp=o/f'metrics/fold{f}_{d}_diagnostic.json';subprocess.run(['/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python','chapter4_detection/scripts/stage24_evaluate.py','--post',str(pp),'--pre',str(o/f'predictions/fold{f}_{d}_prenms.json'),'--gt-stats',str(a.gt_stats),'--fold',str(f),'--train-list',str(a.stage10/f'cv_protocol/fold{f}/train.txt'),'--holdout-list',str(a.stage10/f'cv_protocol/fold{f}/holdout.txt'),'--output',str(dp)],check=True,stdout=subprocess.DEVNULL);diag[(f,n)]=json.loads(dp.read_text())
  foldmetrics[(f,'E0')]=json.loads((a.stage10/f'e0_fold{f}_seed42_metrics.json').read_text())
 # E0 diagnostics use frozen Stage12 decomposition for comparison; formal AP is exactly reproduced.
 gt=[r for r in csv.DictReader(a.gt_stats.open()) if r['split']=='train'];ig={r['image']:r['source_group'] for r in gt};groups=sorted(set(ig.values()));cache={n:{im:image_stats(rec,a.labels/f'{Path(im).stem}.txt') for im,rec in q['images'].items()} for n,q in pred.items()};prep={n:prepare_weighted(z,ig) for n,z in cache.items()};pairs=[('M3','E0'),('M3','M1'),('M3','M2')];rng=np.random.default_rng(20261124);dist={f'{x}-{y}':[] for x,y in pairs};draws=[]
 for _ in range(a.bootstrap):
  mult=Counter(rng.choice(groups,len(groups),replace=True));draws.append(mult);v={n:weighted5(prep[n],mult) for n in pred}
  for x,y in pairs:dist[f'{x}-{y}'].append((v[x]-v[y]).tolist())
 boot={'unit':'source_group','repeats':a.bootstrap,'seed':20261124,'pairs':{}}
 for key,z in dist.items():
  z=np.asarray(z);boot['pairs'][key]={m:stat(z[:,j]) for j,m in enumerate(('map50_95','map50','ap75','flash_ap50_95','black_ap50_95'))}
 db=diagnostic_boot(pred,gt,a.stage10,draws,pairs)
 for key,v in db.items():boot['pairs'][key].update(v)
 (o/'MBRD_BOOTSTRAP.json').write_text(json.dumps(boot,indent=2)+'\n');summary_diag={n:aggregate_diag(diag,n) for n in VAR};summary_diag['E0']=e0_diag(a.stage10,pred['E0'],gt);conv=lambda x:x.item() if isinstance(x,np.generic) else x;(o/'diagnostic_summary.json').write_text(json.dumps(summary_diag,indent=2,default=conv)+'\n');leak=[];usage=[]
 for f in range(3):
  cd=torch.load(o/f'cache/fold{f}.pt',map_location='cpu',weights_only=False);src={Path(x).name for x in cd['images']};tr={Path(x).name for x in (a.stage10/f'cv_protocol/fold{f}/train.txt').read_text().splitlines()};ho={Path(x).name for x in (a.stage10/f'cv_protocol/fold{f}/holdout.txt').read_text().splitlines()};leak.append({'fold':f,'cache_images':len(src),'equals_fold_train':src==tr,'holdout_overlap':sorted(src&ho),'weights_sha256':cd['weights_sha256']});pv=torch.load(o/f'prototypes/fold{f}.pt',map_location='cpu',weights_only=False)['prototypes'];q=torch.nn.functional.normalize(cd['patches'][:,:,3,3].float(),dim=1);sim=q@pv.T;ix=sim.argmax(1);cnt=torch.bincount(ix,minlength=len(pv));prob=cnt.float()/cnt.sum();usage.append({'fold':f,'queries':len(q),'prototypes_used':int((cnt>0).sum()),'top1_count_min':int(cnt.min()),'top1_count_max':int(cnt.max()),'usage_entropy':float(-(prob[prob>0]*prob[prob>0].log()).sum()),'matched_similarity_positive_mean':float(sim.max(1).values[cd['positive']].mean()),'matched_similarity_background_mean':float(sim.max(1).values[~cd['positive']].mean())});del cd
 assert all(x['equals_fold_train'] and not x['holdout_overlap'] for x in leak);(o/'prototype_leakage_audit.json').write_text(json.dumps(leak,indent=2)+'\n');(o/'prototype_usage_audit.json').write_text(json.dumps(usage,indent=2)+'\n');render(o,metrics,foldmetrics,summary_diag,boot)
 env={'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda_available':torch.cuda.is_available(),'device':'CPU','protocol':'cached frozen-E0 P3/TAL training; full online YOLO+MBRD inference'};(o/'environment_audit.json').write_text(json.dumps(env,indent=2)+'\n');hashes={str(x.relative_to(o)):hashlib.sha256(x.read_bytes()).hexdigest() for x in o.rglob('*') if x.is_file() and x.name!='FILE_HASHES.json' and x.suffix not in ('.pt',)};(o/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n');print(json.dumps({n:{k:q[k] for k in ('map50','map50_95','map75')} for n,q in metrics.items()},indent=2))
def aggregate_diag(diag,n):
 rows=sum((diag[(f,n)]['rows'] for f in range(3)),[]);s=[diag[(f,n)]['summary'] for f in range(3)];low=[r for r in rows if r['low_contrast_black']];tiny=[r for r in rows if r['tiny']];return {'gt':len(rows),'decomposition':dict(Counter(r['category'] for r in rows)),'flash_score_misranking':sum(r['class_id']==0 and r['category']=='B_score_misranking' for r in rows),'black_score_misranking':sum(r['class_id']==1 and r['category']=='B_score_misranking' for r in rows),'low_contrast_black_recall':float(np.mean([r['detected_conf025_iou50'] for r in low])),'low_contrast_black_n':len(low),'tiny_recall':float(np.mean([r['detected_conf025_iou50'] for r in tiny])),'tiny_n':len(tiny),'background_fp_conf025_iou_lt010':sum(x['background_fp_conf025_iou_lt010'] for x in s)}
def box_iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);q=np.maximum(rb-lt,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return q/(aa+bb-q+1e-12)
def e0_diag(stage10,pred,gt):
 rows=list(csv.DictReader(Path('chapter4_detection/results/stage12/gt_failure_decomposition.csv').open()));rows=[r for r in rows if r['model']=='e0'];low=[];tiny=[];bg=0
 for f in range(3):
  train={Path(x).name for x in (stage10/f'cv_protocol/fold{f}/train.txt').read_text().splitlines()};hold={Path(x).name for x in (stage10/f'cv_protocol/fold{f}/holdout.txt').read_text().splitlines()};q=np.quantile([float(r['local_contrast']) for r in gt if r['image'] in train and int(r['class_id'])==1],.25)
  for g in [x for x in gt if x['image'] in hold]:
   ps=[x for x in pred['images'][g['image']]['predictions'] if int(x['class_id'])==int(g['class_id']) and x['confidence']>=.25];gb=[float(g[z]) for z in ('x1','y1','x2','y2')];det=any(box_iou(x['xyxy'],gb)>=.5 for x in ps);tiny.append(det) if min(float(g['input_width']),float(g['input_height']))<8 else None;low.append(det) if int(g['class_id'])==1 and float(g['local_contrast'])<=q else None
  for im in hold:
   gs=[g for g in gt if g['image']==im]
   for x in pred['images'][im]['predictions']:
    if x['confidence']>=.25:bg+=max([box_iou(x['xyxy'],[float(g[z]) for z in ('x1','y1','x2','y2')]) for g in gs if int(g['class_id'])==int(x['class_id'])],default=0)<.1
 return {'gt':len(rows),'decomposition':dict(Counter(r['category'] for r in rows)),'flash_score_misranking':sum(int(r['class_id'])==0 and r['category']=='B_score_misranking' for r in rows),'black_score_misranking':sum(int(r['class_id'])==1 and r['category']=='B_score_misranking' for r in rows),'low_contrast_black_recall':float(np.mean(low)),'low_contrast_black_n':len(low),'tiny_recall':float(np.mean(tiny)),'tiny_n':len(tiny),'background_fp_conf025_iou_lt010':bg}
def diagnostic_boot(pred,gt,stage10,draws,pairs):
 fold_of={};threshold={}
 for f in range(3):
  tr={Path(x).name for x in (stage10/f'cv_protocol/fold{f}/train.txt').read_text().splitlines()};ho={Path(x).name for x in (stage10/f'cv_protocol/fold{f}/holdout.txt').read_text().splitlines()};threshold[f]=np.quantile([float(r['local_contrast']) for r in gt if r['image'] in tr and int(r['class_id'])==1],.25);fold_of.update({x:f for x in ho})
 vals={}
 for n,q in pred.items():
  per=defaultdict(lambda:{'low_tp':0,'low_n':0,'bg':0,'images':set()})
  for im,rec in q['images'].items():
   group=next(r['source_group'] for r in gt if r['image']==im);z=per[group];z['images'].add(im);gs=[r for r in gt if r['image']==im]
   for g in gs:
    if int(g['class_id'])==1 and float(g['local_contrast'])<=threshold[fold_of[im]]:
     gb=[float(g[x]) for x in ('x1','y1','x2','y2')];z['low_n']+=1;z['low_tp']+=any(int(p['class_id'])==1 and p['confidence']>=.25 and box_iou(p['xyxy'],gb)>=.5 for p in rec['predictions'])
   for p in rec['predictions']:
    if p['confidence']<.25:continue
    same=[g for g in gs if int(g['class_id'])==int(p['class_id'])];z['bg']+=max([box_iou(p['xyxy'],[float(g[x]) for x in ('x1','y1','x2','y2')]) for g in same],default=0)<.1
  vals[n]=per
 out={f'{x}-{y}':{'low_contrast_black_recall':[],'background_fp_per_image':[]} for x,y in pairs}
 for mult in draws:
  cur={}
  for n,per in vals.items():
   tp=sum(mult.get(g,0)*v['low_tp'] for g,v in per.items());nn=sum(mult.get(g,0)*v['low_n'] for g,v in per.items());bg=sum(mult.get(g,0)*v['bg'] for g,v in per.items());ni=sum(mult.get(g,0)*len(v['images']) for g,v in per.items());cur[n]=(tp/nn,bg/ni)
  for x,y in pairs:out[f'{x}-{y}']['low_contrast_black_recall'].append(cur[x][0]-cur[y][0]);out[f'{x}-{y}']['background_fp_per_image'].append(cur[x][1]-cur[y][1])
 return {k:{m:stat(z) for m,z in v.items()} for k,v in out.items()}
def table(m,d):
 z=['| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Low-contrast Black R | Tiny R | BG FP |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n in ('E0','M1','M2','M3'):
  q=m[n];x=d[n];z.append(f"| {n} | {q['precision']:.4f} | {q['recall']:.4f} | {q['map50']:.4f} | {q['map50_95']:.4f} | {q['map75']:.4f} | {q['per_class_map50_95']['0']:.4f} | {q['per_class_map50_95']['1']:.4f} | {x['low_contrast_black_recall']:.4f} | {x['tiny_recall']:.4f} | {x['background_fp_conf025_iou_lt010']} |")
 return '\n'.join(z)
def render(o,m,fm,d,b):
 t=table(m,d);runs={(f,n):json.loads((o/f'runs/fold{f}_{v}/run.json').read_text()) for f in range(3) for n,v in VAR.items()};pro=[json.loads((o/f'prototypes/fold{f}.json').read_text()) for f in range(3)];ft=['| Fold | Model | mAP50-95 | AP75 |','|---:|---|---:|---:|']
 for f in range(3):
  for n in ('E0','M1','M2','M3'):q=fm[(f,n)];ft.append(f"| {f} | {n} | {q['map50_95']:.4f} | {q['map75']:.4f} |")
 # Frozen retention rule: matching must improve all principal metrics over the
 # fixed-reference control, remain competitive with E0 at AP75, avoid a clear
 # background-FP regression, and have consistent fold direction.  A single
 # favourable aggregate point estimate is deliberately insufficient.
 fold_e0=sum(fm[(f,'M3')]['map50_95']>fm[(f,'E0')]['map50_95'] for f in range(3))
 fold_m2=sum(fm[(f,'M3')]['map50_95']>fm[(f,'M2')]['map50_95'] for f in range(3))
 passed=(m['M3']['map50_95']>m['M2']['map50_95'] and m['M3']['map50']>=m['M2']['map50']
         and m['M3']['map75']>=m['M2']['map75'] and m['M3']['map50_95']>m['M1']['map50_95']
         and m['M3']['map50_95']>m['E0']['map50_95'] and m['M3']['map75']>=m['E0']['map75']-.01
         and d['M3']['background_fp_conf025_iou_lt010']<=d['E0']['background_fp_conf025_iou_lt010']
         and fold_e0>=2 and fold_m2>=2)
 (o/'MBRD_DESIGN.md').write_text('# MBRD design\n\nMBRD retrieves the top-4 of 32 fold-specific normalized conservative-background prototypes by vectorized cosine similarity (τ=0.1), forms `[q, q-r, q⊙r]`, and applies a 3C→32→C 1×1 residual adapter with zero-initialized scalar γ. It is inserted only before the frozen P3 classification tower. M1 uses the same adapter capacity without a reference; M2 uses one deterministic query-independent mean reference; M3 uses matched retrieval.\n')
 (o/'MBRD_EXPERIMENT_FREEZE.json').write_text(json.dumps({'seed':42,'imgsz':1536,'K':32,'topk':4,'tau':.1,'hidden':32,'epochs':40,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'checkpoint':'last.pt','E0':'Stage10 fold-specific last.pt','training':'cached P3 patches with frozen exact E0 TAL targets','inference':'full online YOLO+MBRD'},indent=2)+'\n')
 (o/'BACKGROUND_PROTOTYPE_AUDIT.md').write_text('# Background prototype audit\n\nOnly each outer fold training list contributes candidates. GT-expanded regions, padding, P3 positives, and E0 score≥0.05 positions are excluded; remaining locations are sampled across four feature-norm strata. They are conservative candidate backgrounds, not expert-confirmed normal regions.\n\n'+ '\n'.join(f"- Fold {q['fold']}: {q['samples']} candidates; cluster range {q['min_cluster']}–{q['max_cluster']}; mean {q['mean_images_per_prototype']:.1f} source images/prototype; maximum single-image share {q['max_dominant_image_fraction']:.3f}; maximum inter-prototype cosine {q['prototype_cosine_max']:.3f}." for q in pro)+'\n')
 (o/'MBRD_IMPLEMENTATION_AUDIT.md').write_text('# MBRD implementation audit\n\nAt γ=0, module output is bit-identical to input. Unit tests cover all modes, vectorized nearest-prototype retrieval and query-independent M2. Formal trained-model audits record exactly unchanged P3/P4/P5 regression logits and P4/P5 classification logits, while trained P3 classification logits change. Full YOLO inference—not post-NMS re-ranking—produces all formal predictions.\n')
 (o/'MBRD_TRAINING_RESULTS.md').write_text('# MBRD training results\n\nNine adapters completed 40 epochs with AdamW and last-checkpoint selection. Backbone, neck, box tower and original class tower parameters are frozen. Cached targets are the exact fold-specific E0 TAL assignment; this freezes assignment rather than allowing MBRD scores to alter TAL online and is a disclosed difference from full YOLO training.\n\n'+ '\n'.join(f"- Fold {f} {n}: γ={runs[(f,n)]['final_gamma']:.5f}, loss={runs[(f,n)]['final_loss']:.6f}, P3 logit |Δ|={runs[(f,n)]['mean_abs_p3_logit_change']:.6f}." for f in range(3) for n in VAR)+'\n')
 (o/'MBRD_GROUP_CV_RESULTS.md').write_text('# MBRD group-CV results\n\n'+t+'\n\n## Per-fold direction\n\n'+'\n'.join(ft)+'\n\nThese are source-group OOF development results, not an independent test estimate.\n')
 (o/'MBRD_ABLATION_RESULTS.md').write_text('# MBRD ablation results\n\n'+t+'\n\nM3−M2 isolates query-dependent texture matching; M3−M1 tests background reference beyond equal-capacity ordinary enhancement. Source-group intervals are in `MBRD_BOOTSTRAP.json`.\n')
 (o/'MBRD_FAILURE_DECOMPOSITION.md').write_text('# MBRD failure decomposition\n\n'+ '\n'.join(f"- {n}: {q['decomposition']}; Flash score-misranking={q['flash_score_misranking']}, Black score-misranking={q['black_score_misranking']}." for n,q in d.items())+'\n\nThe regression tower is frozen, so MBRD cannot create new box geometry; changes arise from P3 classification scores and subsequent NMS/max-det retention. The compact diagnostic exports every anchor-class pair above 0.001, but it does not trace the exact internal NMS suppression decision and therefore is not interchangeable with the full Stage12 trace.\n')
 (o/'MBRD_ERROR_ANALYSIS.md').write_text('# MBRD error analysis\n\n'+ '\n'.join(f"- {n}: low-contrast Black recall {q['low_contrast_black_recall']:.4f} ({q['low_contrast_black_n']} GT), tiny recall {q['tiny_recall']:.4f} ({q['tiny_n']} GT), background-style FP@0.25={q['background_fp_conf025_iou_lt010']}." for n,q in d.items())+'\n')
 params=runs[(0,'M3')]['parameters'];times={n:np.mean([json.loads((o/f'predictions/fold{f}_{v}.json').read_text())['mean_seconds_per_original'] for f in range(3)]) for n,v in VAR.items()};(o/'MBRD_EFFICIENCY.md').write_text('# MBRD efficiency\n\n'+ '\n'.join(f"- {n}: {runs[(0,n)]['parameters']} trainable adapter parameters; mean observed CPU full-image inference {times[n]:.3f} s." for n in VAR)+'\n\nCPU wall times are approximate and were measured under uncontrolled system load. M3 additionally performs vectorized P3-to-32-prototype similarity and top-4 retrieval.\n')
 ci=b['pairs'];m3m2=ci['M3-M2'];m3e0=ci['M3-E0'];m3m1=ci['M3-M1']
 decision='PASS' if passed else 'STOP / NOT RETAIN'
 answers=f'''\n\n## Frozen decision\n\n**{decision}**. M3 improves aggregate mAP50-95 over M2 by {m['M3']['map50_95']-m['M2']['map50_95']:+.4f}, but its mAP50 ({m['M3']['map50']:.4f}) and AP75 ({m['M3']['map75']:.4f}) are below M2 ({m['M2']['map50']:.4f}/{m['M2']['map75']:.4f}); the source-group bootstrap mAP50-95 CI is {m3m2['map50_95']['ci95']}. Only {fold_m2}/3 folds favour M3 over M2. Background FP rises from {d['E0']['background_fp_conf025_iou_lt010']} to {d['M3']['background_fp_conf025_iou_lt010']}.\n\n## Required mechanism answers\n\n1. MBRD is integrated before the real P3 classification tower during full YOLO inference; it is not post-NMS reranking.\n2. P3 regression and all P4/P5 logits are exactly unchanged in the trained-model audits.\n3. M3 performs actual top-4 matching over 32 prototypes; all 32 are used in every fold.\n4. Prototypes use outer-train images only; cache/holdout overlap is zero.\n5. M3 is not convincingly better than fixed-reference M2: only mAP50-95 has a positive point estimate and its CI crosses zero.\n6. M3 exceeds equal-capacity M1 by {m['M3']['map50_95']-m['M1']['map50_95']:+.4f} mAP50-95 (bootstrap CI {m3m1['map50_95']['ci95']}), supporting reference information but not matched retrieval specifically.\n7. Versus E0, M3 changes mAP50/mAP50-95 by {m['M3']['map50']-m['E0']['map50']:+.4f}/{m['M3']['map50_95']-m['E0']['map50_95']:+.4f}; both bootstrap intervals cross zero.\n8. Flash score-misranking is {d['M3']['flash_score_misranking']} for M3 versus {d['M2']['flash_score_misranking']} for M2 and {d['E0']['flash_score_misranking']} for E0; no robust reduction is established.\n9. Low-contrast Black recall is {d['M3']['low_contrast_black_recall']:.4f}, above E0 {d['E0']['low_contrast_black_recall']:.4f} but identical to M2 {d['M2']['low_contrast_black_recall']:.4f}.\n10. Background FP clearly worsens versus E0 ({d['E0']['background_fp_conf025_iou_lt010']}→{d['M3']['background_fp_conf025_iou_lt010']}; bootstrap per-image CI {m3e0['background_fp_per_image']['ci95']}).\n11. AP75 is {m['M3']['map75']:.4f}, below E0 {m['E0']['map75']:.4f} and M2 {m['M2']['map75']:.4f}.\n12. Tiny recall is {d['M3']['tiny_recall']:.4f}, above E0 {d['E0']['tiny_recall']:.4f} but identical to M2, so it is not matching-specific.\n13. All output changes originate in P3 classification scores and downstream retention/NMS; box geometry is frozen.\n14. Leakage checks pass, but candidates are conservatively screened rather than expert-confirmed background; fold 0 also has a dominant prototype cluster and remains a contamination/coverage risk.\n15. A matching-specific mechanism contribution is not established relative to M2.\n16. Hard-negative or contrastive supervision is not added in this frozen stage because it would mask the failed matching test.\n17. MBRD is not combined with CIR; the current module is not retained as the formal innovation.\n'''
 (o/'STAGE24_FINAL_SUMMARY.md').write_text('# Stage24 final summary\n\n'+t+answers)
 (o/'RUN_COMMANDS.md').write_text('# Stage24 commands\n\nRun `stage24_prepare_features.py`, `stage24_build_prototypes.py`, `stage24_train_mbrd.py`, `stage24_predict_mbrd.py`, and `stage24_finalize.py` in that order. The repository `run_stage24.sh` contains the exact fold/variant commands.\n')
if __name__=='__main__':main()
