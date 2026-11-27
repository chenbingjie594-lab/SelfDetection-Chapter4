#!/usr/bin/env python3
"""Stage26 OOF metrics, proposal coverage, paired bootstrap, and reports."""
import argparse,csv,hashlib,json,platform,subprocess
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch,ultralytics
from chapter4_detection.scripts.stage24_finalize import weighted5,stat,box_iou
from chapter4_detection.scripts.paired_group_bootstrap import image_stats,prepare_weighted
from chapter4_detection.scripts.stage25_evaluate import classify,THR
from chapter4_detection.scripts.stage25_finalize import froc_by_group,fixed_fppi
MODELS=('E0','F0','F1','F2');PAIRS=(('F2','E0'),('F2','F0'),('F2','F1'))
def best_iou(rec,g):
 b=[float(g[x]) for x in ('x1','y1','x2','y2')];return max([box_iou(p['xyxy'],b) for p in rec['predictions'] if int(p['class_id'])==int(g['class_id'])],default=0.)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('chapter4_detection/results/stage26'));p.add_argument('--stage10',type=Path,default=Path('chapter4_detection/results/stage10'));p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args();o=a.root;pred={'E0':json.loads((a.stage10/'e0_oof_seed42_predictions.json').read_text())};ind={}
 for v in ('F0','F1','F2'):
  ims={};ii={};secs=0
  for f in range(3):
   q=json.loads((o/f'predictions/fold{f}_{v}.json').read_text());z=json.loads((o/f'predictions/fold{f}_{v}_independent.json').read_text());ims.update(q['images']);ii.update(z['images']);secs+=q['mean_seconds_per_original']*q['images_count']
  pred[v]={'schema_version':1,'model':f'E0+{v}','images_count':len(ims),'images':ims,'mean_seconds_per_original':secs/len(ims)};ind[v]={'schema_version':1,'model':v,'images_count':len(ii),'images':ii};(o/f'{v.lower()}_oof_predictions.json').write_text(json.dumps(pred[v])+'\n');(o/f'{v.lower()}_independent_oof.json').write_text(json.dumps(ind[v])+'\n')
 metrics={};(o/'metrics').mkdir(exist_ok=True)
 for n,q in pred.items():
  pp=a.stage10/'e0_oof_seed42_predictions.json' if n=='E0' else o/f'{n.lower()}_oof_predictions.json';mp=o/f'metrics/{n.lower()}_oof.json';subprocess.run(['/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python','chapter4_detection/scripts/evaluate_original.py','--predictions',str(pp),'--labels',str(a.labels),'--output',str(mp)],check=True,stdout=subprocess.DEVNULL);metrics[n]=json.loads(mp.read_text())
 gt=[r for r in csv.DictReader(a.gt_stats.open()) if r['split']=='train'];fail={(r['image'],int(r['gt_id'])):r['category'] for r in csv.DictReader(Path('chapter4_detection/results/stage12/gt_failure_decomposition.csv').open()) if r['model']=='e0'};groups=sorted(set(r['source_group'] for r in gt));ig={r['image']:r['source_group'] for r in gt};rows=[]
 for g in gt:
  row={'image':g['image'],'gt_id':int(g['gt_id']),'source_group':g['source_group'],'class_id':int(g['class_id']),'tiny':min(float(g['input_width']),float(g['input_height']))<8,'stage12_A':fail.get((g['image'],int(g['gt_id'])))=='A_generation_insufficient'}
  for n in MODELS:row[n]=best_iou(pred[n]['images'][g['image']],g)
  for n in ('F0','F1','F2'):row[n+'_independent']=best_iou(ind[n]['images'][g['image']],g)
  rows.append(row)
 (o/'coverage_rows.json').write_text(json.dumps(rows,indent=2)+'\n');taxonomy={n:classify(q['images'],gt,.25) for n,q in pred.items()};(o/'background_error_taxonomy.json').write_text(json.dumps(taxonomy,indent=2)+'\n');coverage={}
 for n in MODELS:
  coverage[n]={f'coverage{t}':float(np.mean([r[n]>=t/100 for r in rows])) for t in (50,75,90)};coverage[n]['tiny_coverage75']=float(np.mean([r[n]>=.75 for r in rows if r['tiny']]))
 for n in ('F0','F1','F2'):
  coverage[n]['independent_coverage75']=float(np.mean([r[n+'_independent']>=.75 for r in rows]));coverage[n]['stage12_A_recovered75']=int(sum(r['stage12_A'] and r['E0']<.75 and r[n]>=.75 for r in rows));coverage[n]['tiny_new_recovered75']=int(sum(r['tiny'] and r['E0']<.75 and r[n]>=.75 for r in rows))
 cand={}
 for n,q in ind.items():
  total=good50=good75=bg=0
  for im,rec in q['images'].items():
   gs=[g for g in gt if g['image']==im]
   for x in rec['predictions']:
    total+=1;same=max([box_iou(x['xyxy'],[float(g[z]) for z in ('x1','y1','x2','y2')]) for g in gs if int(g['class_id'])==int(x['class_id'])],default=0);anyv=max([box_iou(x['xyxy'],[float(g[z]) for z in ('x1','y1','x2','y2')]) for g in gs],default=0);good50+=same>=.5;good75+=same>=.75;bg+=anyv<.1
  cand[n]={'total':int(total),'same_class_iou50':int(good50),'same_class_iou75':int(good75),'background_style':int(bg)}
 (o/'candidate_recovery.json').write_text(json.dumps({'coverage':coverage,'candidates':cand},indent=2)+'\n')
 prep={n:prepare_weighted({im:image_stats(rec,a.labels/f'{Path(im).stem}.txt') for im,rec in q['images'].items()},ig) for n,q in pred.items()};per={n:froc_by_group(q,gt,groups) for n,q in pred.items()};rng=np.random.default_rng(20261127);dist={f'{x}-{y}':defaultdict(list) for x,y in PAIRS}
 bygroup=defaultdict(list)
 for r in rows:bygroup[r['source_group']].append(r)
 for _ in range(a.bootstrap):
  mult=Counter(rng.choice(groups,len(groups),replace=True));ap={n:weighted5(prep[n],mult) for n in pred};fr={n:fixed_fppi(per[n],mult,.1) for n in pred}
  for x,y in PAIRS:
   for j,k in enumerate(('map50_95','map50','ap75','flash_ap50_95','black_ap50_95')):dist[f'{x}-{y}'][k].append(float(ap[x][j]-ap[y][j]))
   den=sum(mult.get(g,0)*len(bygroup[g]) for g in groups);tinyden=sum(mult.get(g,0)*sum(r['tiny'] for r in bygroup[g]) for g in groups);dist[f'{x}-{y}']['coverage75'].append(sum(mult.get(g,0)*sum(int(r[x]>=.75)-int(r[y]>=.75) for r in bygroup[g]) for g in groups)/den);dist[f'{x}-{y}']['tiny_coverage75'].append(sum(mult.get(g,0)*sum((int(r[x]>=.75)-int(r[y]>=.75)) for r in bygroup[g] if r['tiny']) for g in groups)/tinyden);dist[f'{x}-{y}']['recall_at_fppi_0.1'].append(fr[x]-fr[y])
 boot={'unit':'source_group','repeats':a.bootstrap,'seed':20261127,'pairs':{k:{m:stat(v) for m,v in z.items()} for k,z in dist.items()}};(o/'FSPR_BOOTSTRAP.json').write_text(json.dumps(boot,indent=2)+'\n');render(o,metrics,coverage,cand,taxonomy,boot)
 env={'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cuda_available':torch.cuda.is_available(),'device':'CPU','protocol':'frozen E0 + independently trained dense stride-4 sidecar'};(o/'environment_audit.json').write_text(json.dumps(env,indent=2)+'\n');hashes={str(x.relative_to(o)):hashlib.sha256(x.read_bytes()).hexdigest() for x in o.rglob('*') if x.is_file() and x.name!='FILE_HASHES.json' and x.suffix!='.pt'};(o/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n');print(table(metrics,coverage,taxonomy))
def table(m,c,t):
 z=['| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP | Coverage75 | Tiny Coverage75 | BG FP |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for n in MODELS:q=m[n];z.append(f"| {n} | {q['precision']:.4f} | {q['recall']:.4f} | {q['map50']:.4f} | {q['map50_95']:.4f} | {q['map75']:.4f} | {q['per_class_map50_95']['0']:.4f} | {q['per_class_map50_95']['1']:.4f} | {c[n]['coverage75']:.4f} | {c[n]['tiny_coverage75']:.4f} | {t[n]['counts'].get('background_style_prediction',0)} |")
 return '\n'.join(z)
def render(o,m,c,can,t,b):
 tab=table(m,c,t);runs={(f,v):json.loads((o/f'runs/fold{f}_{v}/run.json').read_text()) for f in range(3) for v in ('F0','F1','F2')};times={v:np.mean([json.loads((o/f'predictions/fold{f}_{v}_audit.json').read_text())['mean_cpu_seconds_full_e0_plus_fspr'] for f in range(3)]) for v in ('F0','F1','F2')}
 (o/'FSPR_DESIGN.md').write_text('# FSPR design\n\nFSPR is a 9,032-parameter frozen-E0 sidecar. It projects backbone P2 and neck P3 to 24 channels, nearest-aligns P3 at stride 4, optionally adds the explicit within-2×2 phase residual and fixed phase coordinates, and predicts two heatmap classes, bounded center offsets and four positive edge distances. F0/F1/F2 have identical capacity through zero-input controls. Original Detect logits, boxes and NMS outputs are never modified.\n')
 (o/'FSPR_EXPERIMENT_FREEZE.json').write_text(json.dumps({'seed':42,'folds':'Stage10 source-group 3-fold','imgsz':1536,'epochs':40,'batch':256,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'checkpoint':'last.pt','positive_radius':1,'score_min':.05,'topk_per_class':100,'fspr_nms':.7,'cross_source_iou':.7,'loss':'focal + IoU + offset + .5 edge'},indent=2)+'\n')
 (o/'FSPR_TRAINING_RESULTS.md').write_text('# FSPR training results\n\nAll nine CPU cached-feature runs completed 40 epochs with non-zero required gradients for applicable branches and no holdout early stopping. F0 intentionally has zero P2 gradient because its P2 inputs are the frozen zero ablation.\n\n'+'\n'.join(f"- Fold {f} {v}: loss={runs[(f,v)]['final']['loss']:.4f}, IoU={runs[(f,v)]['final']['iou']:.4f}, offset={runs[(f,v)]['final']['offset']:.4f}, edge={runs[(f,v)]['final']['edge']:.4f}." for f in range(3) for v in ('F0','F1','F2'))+'\n')
 (o/'FSPR_GROUP_CV_RESULTS.md').write_text('# FSPR group-CV results\n\n'+tab+'\n\nThese are repeatedly inspected OOF development results, not an independent test estimate.\n')
 (o/'FSPR_ABLATION_RESULTS.md').write_text('# FSPR ablation results\n\n'+tab+'\n\nF1−F0 isolates P2 features; F2−F1 isolates the explicit phase residual; F2−E0 tests the deployed merged system. Paired intervals are in `FSPR_BOOTSTRAP.json`.\n')
 (o/'FSPR_CANDIDATE_RECOVERY.md').write_text('# FSPR candidate recovery\n\n'+'\n'.join(f"- {v}: {can[v]['total']} independent proposals; IoU≥.50 candidates={can[v]['same_class_iou50']}; IoU≥.75={can[v]['same_class_iou75']}; background-style={can[v]['background_style']}; Stage12-A GT newly covered@.75={c[v]['stage12_A_recovered75']}; tiny GT newly covered@.75={c[v]['tiny_new_recovered75']}." for v in ('F0','F1','F2'))+'\n\nCoverage is offline GT-oracle analysis only; GT is never used in proposal generation or merge.\n')
 (o/'FSPR_TINY_DEFECT_ANALYSIS.md').write_text('# FSPR tiny-defect analysis\n\n'+ '\n'.join(f"- {n}: tiny Coverage75={c[n]['tiny_coverage75']:.4f}." for n in MODELS)+'\n')
 (o/'FSPR_BACKGROUND_ERROR_ANALYSIS.md').write_text('# FSPR background error analysis\n\nThe Stage25 any-class-IoU<0.10 definition is used uniformly.\n\n'+'\n'.join(f"- {n}: {t[n]['counts']}." for n in MODELS)+'\n')
 (o/'FSPR_EFFICIENCY.md').write_text('# FSPR efficiency\n\nAll variants add 9,032 parameters. Observed concurrent CPU E0+FSPR full-image times: '+', '.join(f'{v}={times[v]:.3f}s' for v in times)+'. GPU memory is unavailable.\n')
 decision='RETAIN' if m['F2']['map50_95']>m['E0']['map50_95'] and m['F2']['map50_95']>m['F1']['map50_95'] and c['F2']['coverage75']>c['F1']['coverage75'] and t['F2']['counts'].get('background_style_prediction',0)<500 else 'STOP / NOT RETAIN'
 (o/'STAGE26_FINAL_SUMMARY.md').write_text('# Stage26 final summary\n\n'+tab+f'\n\n## Decision\n\n**{decision}**. The decision requires deployable detection improvement, not oracle coverage alone. Detailed candidate counts and uncertainty are preserved in the Stage26 reports. CIR combination is deferred unless FSPR is retained independently.\n')
 (o/'RUN_COMMANDS.md').write_text('# Stage26 commands\n\nUse `stage26_prepare_features.py`, `stage26_train_fspr.py`, `stage26_patch_equivalence.py`, `stage26_predict_fspr.py`, and `stage26_finalize.py` in that order. Exact commands are in `run_stage26.sh`.\n')
if __name__=='__main__':main()
