"""Shared Stage23 evaluation and quality-gate helpers."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np,torch

def iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);inter=np.maximum(rb-lt,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return float(inter/(aa+bb-inter+1e-12))
def torch_iou(a,b):
 lt=torch.maximum(a[:,:2],b[:,:2]);rb=torch.minimum(a[:,2:],b[:,2:]);inter=(rb-lt).clamp_min(0).prod(1);aa=(a[:,2:]-a[:,:2]).clamp_min(0).prod(1);bb=(b[:,2:]-b[:,:2]).clamp_min(0).prod(1);return inter/(aa+bb-inter+1e-9)
def load_gt(path,allowed=None):
 rows=list(csv.DictReader(Path(path).open()));return [r for r in rows if allowed is None or r['image'] in allowed]
def best_rows(base,refined,gt_rows):
 """Re-select the maximum-IoU same-class candidate independently after refinement."""
 base=json.loads(Path(base).read_text())['images'] if not isinstance(base,dict) else base
 refined=json.loads(Path(refined).read_text())['images'] if not isinstance(refined,dict) else refined
 out=[]
 for g in gt_rows:
  name=g['image'];cls=int(g['class_id']);gb=[float(g[x]) for x in ('x1','y1','x2','y2')]
  p0=[p for p in base.get(name,{}).get('predictions',[]) if int(p['class_id'])==cls];p1=[p for p in refined.get(name,{}).get('predictions',[]) if int(p['class_id'])==cls]
  v0=[iou(p['xyxy'],gb) for p in p0];v1=[iou(p['xyxy'],gb) for p in p1];k0=int(np.argmax(v0)) if v0 else -1;k1=int(np.argmax(v1)) if v1 else -1
  fixed=v1[k0] if 0<=k0<len(v1) else 0.;q0=v0[k0] if k0>=0 else 0.;q1=v1[k1] if k1>=0 else 0.
  out.append({'image':name,'gt_id':int(g['gt_id']),'source_group':g['source_group'],'class_id':cls,'tiny':min(float(g['input_width']),float(g['input_height']))<8,'iou_before':q0,'fixed_candidate_iou_after':fixed,'best_iou_after':q1,'best_id_before':k0,'best_id_after':k1,'best_id_switched':k0!=k1,'gain75':q0<.75<=q1,'lose75':q0>=.75>q1})
 return out
def summarize(rows):
 def cov(z,key):return sum(r[key]>=.75 for r in z)
 tiny=[r for r in rows if r['tiny']];flash=[r for r in rows if r['class_id']==0];black=[r for r in rows if r['class_id']==1]
 return {'gt':len(rows),'mean_best_iou_before':float(np.mean([r['iou_before'] for r in rows])),'mean_fixed_candidate_iou_after':float(np.mean([r['fixed_candidate_iou_after'] for r in rows])),'mean_best_iou_after':float(np.mean([r['best_iou_after'] for r in rows])),'coverage75_before_count':cov(rows,'iou_before'),'coverage75_after_count':cov(rows,'best_iou_after'),'tiny_gt':len(tiny),'tiny_coverage75_before_count':cov(tiny,'iou_before'),'tiny_coverage75_after_count':cov(tiny,'best_iou_after'),'flash_coverage75_after_count':cov(flash,'best_iou_after'),'black_coverage75_after_count':cov(black,'best_iou_after'),'gain75':sum(r['gain75'] for r in rows),'lose75':sum(r['lose75'] for r in rows),'best_id_switches':sum(r['best_id_switched'] for r in rows)}
