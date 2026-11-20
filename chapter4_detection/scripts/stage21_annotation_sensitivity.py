#!/usr/bin/env python3
"""Evaluate frozen E0 OOF predictions against temporary perturbed GT copies."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np, torch
from ultralytics.utils.metrics import DetMetrics,box_iou
from chapter4_detection.scripts.evaluate_original import match_predictions

def iou(a,b):
 lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);q=np.maximum(rb-lt,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return q/(aa+bb-q+1e-12)
def perturb(b,op,d,rng,w=4096,h=1024):
 z=np.array(b,float)
 if op=='center':
  dx,dy=rng.uniform(-d,d,2);z[[0,2]]+=dx;z[[1,3]]+=dy
 elif op=='size':
  dx,dy=rng.uniform(-d,d,2);z[0]-=dx;z[2]+=dx;z[1]-=dy;z[3]+=dy
 elif op.startswith('single_'):
  j={'single_l':0,'single_t':1,'single_r':2,'single_b':3}[op];z[j]+=rng.choice([-d,d])
 elif op=='independent':z+=rng.uniform(-d,d,4)
 z[[0,2]]=np.clip(z[[0,2]],0,w);z[[1,3]]=np.clip(z[[1,3]],0,h)
 if z[2]<=z[0]:z[2]=min(w,z[0]+1e-3)
 if z[3]<=z[1]:z[3]=min(h,z[1]+1e-3)
 return z
def evaluate(pred,gtrows,boxes):
 m=DetMetrics(names={0:'Flash point',1:'Big black spots'});cov=[]
 by={}
 for r,b in zip(gtrows,boxes):by.setdefault(r['image'],[]).append((r,b))
 for ii,(name,items) in enumerate(sorted(by.items())):
  ps=pred[name]['predictions'];pc=torch.tensor([p['class_id'] for p in ps],dtype=torch.float32);pb=torch.tensor([p['xyxy'] for p in ps],dtype=torch.float32).reshape(-1,4);cf=torch.tensor([p['confidence'] for p in ps],dtype=torch.float32)
  tc=torch.tensor([int(r['class_id']) for r,b in items],dtype=torch.float32);tb=torch.tensor([b for r,b in items],dtype=torch.float32).reshape(-1,4);mat=box_iou(tb,pb) if len(ps) else torch.zeros((len(items),0))
  cr=match_predictions(pc,tc,mat) if len(ps) else torch.zeros((0,10),dtype=torch.bool)
  m.stats['tp'].append(cr.numpy());m.stats['conf'].append(cf.numpy());m.stats['pred_cls'].append(pc.numpy());m.stats['target_cls'].append(tc.numpy());m.stats['target_img'].append(np.full(len(tc),ii))
  for j,(r,b) in enumerate(items):
   best=max((float(mat[j,k]) for k,p in enumerate(ps) if int(p['class_id'])==int(r['class_id'])),default=0);cov.append((best>=.75,str(r['tiny'])=='True'))
 m.process(plot=False)
 return {'map50':float(m.box.map50),'map75':float(m.box.map75),'map50_95':float(m.box.map),'flash_ap':float(m.maps[0]),'black_ap':float(m.maps[1]),'coverage75':float(np.mean([x[0] for x in cov])),'tiny_coverage75':float(np.mean([x[0] for x in cov if x[1]]))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 pred=json.loads(a.predictions.read_text())['images'];gt=[r for r in csv.DictReader(a.gt_stats.open()) if r['split']=='train'];base=[np.array([float(r[x]) for x in ('x1','y1','x2','y2')]) for r in gt]
 # Freeze tiny definition at the Stage10 input-scale short side <8 px.
 for r in gt:r['tiny']=str(min(float(r['input_width']),float(r['input_height']))<8)
 rows=[]
 for op,d,seeds in [('baseline',0,[0])]+[(op,d,range(10)) for d in (.5,1,2) for op in ('center','size','single_l','single_t','single_r','single_b','independent')]:
  for seed in seeds:
   rng=np.random.default_rng(21000+int(d*10)*100+seed+sum(map(ord,op)));boxes=base if op=='baseline' else [perturb(b,op,d,rng) for b in base];q=evaluate(pred,gt,boxes)
   rows.append({'operation':op,'delta_original_px':d,'seed':seed,'valid_boxes':all(b[2]>b[0] and b[3]>b[1] for b in boxes),'gt_count':len(boxes),**q})
 with (a.output/'annotation_perturbation_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 baseq=rows[0];summary={'protocol':{'predictions':'frozen Stage10 E0 OOF','random_seeds':list(range(10)),'coordinate_system':'4096x1024 original','gt_count':len(gt)},'baseline':baseq,'conditions':{}}
 for d in (.5,1,2):
  for op in ('center','size','single_l','single_t','single_r','single_b','independent'):
   z=[r for r in rows if r['operation']==op and r['delta_original_px']==d];summary['conditions'][f'{op}_{d}']={k:{'mean':float(np.mean([r[k] for r in z])),'sd':float(np.std([r[k] for r in z],ddof=1)),'delta_mean':float(np.mean([r[k] for r in z])-baseq[k])} for k in ('map50','map75','map50_95','flash_ap','black_ap','coverage75','tiny_coverage75')}
 (a.output/'annotation_sensitivity_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
