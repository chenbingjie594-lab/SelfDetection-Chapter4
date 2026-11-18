#!/usr/bin/env python3
"""Analyze final and oracle pre-NMS DFL candidates on Stage10 OOF images."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
EDGES=('left','top','right','bottom')
def ious(boxes,b):
 x1=np.maximum(boxes[:,0],b[0]);y1=np.maximum(boxes[:,1],b[1]);x2=np.minimum(boxes[:,2],b[2]);y2=np.minimum(boxes[:,3],b[3]);inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1);a=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1]);bb=(b[2]-b[0])*(b[3]-b[1]);return inter/(a+bb-inter+1e-12)
def gt(label,W,H):
 out=[]
 for i,line in enumerate(label.read_text().splitlines()):
  z=line.split();c=int(z[0]);x,y,w,h=map(float,z[1:]);out.append((i,c,np.array([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H])))
 return out
def softmax(x):x=x-x.max(1,keepdims=True);q=np.exp(x);return q/q.sum(1,keepdims=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage10',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);rows=[]
 for model in ('e0','cbcd'):
  for fold in range(3):
   raw=a.raw_root/f'{model}_fold{fold}';man=json.loads((raw/'manifest.json').read_text());formal=json.loads((a.stage10/f'{model}_fold{fold}_seed42_predictions.json').read_text())['images']
   for name,rec in man['records'].items():
    z=np.load(raw/rec['file']);dec=z['decoded'];dfl=z['raw_boxes'].T.reshape(-1,4,16);prob=softmax(dfl.reshape(-1,16)).reshape(-1,4,16);exp=(prob*np.arange(16)).sum(2)
    # decoded is xywh in letterbox input coordinates; convert to original xyxy.
    xywh=dec[:4].T;box=np.column_stack((xywh[:,0]-xywh[:,2]/2,xywh[:,1]-xywh[:,3]/2,xywh[:,0]+xywh[:,2]/2,xywh[:,1]+xywh[:,3]/2));gain=rec['ratio_pad'][0];pad=rec['ratio_pad'][1];box[:,[0,2]]=(box[:,[0,2]]-pad[0])/gain[1];box[:,[1,3]]=(box[:,[1,3]]-pad[1])/gain[0]
    score=dec[4:].T;shapes=rec['feature_shapes'];counts=[q[2]*q[3] for q in shapes];ends=np.cumsum(counts);strides=np.concatenate([np.full(n,s) for n,s in zip(counts,(8,16,32))]);level=np.concatenate([np.full(n,f'P{i}') for n,i in zip(counts,(3,4,5))])
    # Anchor centers in input coords, in the same flatten order as Detect.
    anchors=[]
    for q,s in zip(shapes,(8,16,32)):
     H,W=q[2:];yy,xx=np.meshgrid(np.arange(H)+.5,np.arange(W)+.5,indexing='ij');anchors.append(np.column_stack((xx.ravel()*s,yy.ravel()*s)))
    anchors=np.concatenate(anchors)
    W,H=formal[name]['width'],formal[name]['height']
    for gi,c,b in gt(a.labels/f'{Path(name).stem}.txt',W,H):
     ov=ious(box,b);valid=np.where(score[:,c]>=.001)[0];oracle=int(valid[np.argmax(ov[valid])]) if len(valid) else None
     fp=[q for q in formal[name]['predictions'] if int(q['class_id'])==c];final_box=np.array(max(fp,key=lambda q:ious(np.asarray([q['xyxy']]),b)[0])['xyxy']) if fp else None
     final=None
     if final_box is not None:
      # NMS preserves one raw candidate; coordinate nearest-neighbor is exact up to export fp rounding.
      final=int(np.argmin(np.abs(box-final_box).sum(1)+np.where(score[:,c]>=.001,0,1e9)))
     base={'model':model,'fold':fold,'image':name,'gt_id':gi,'class_id':c,'gt_width':b[2]-b[0],'gt_height':b[3]-b[1],'gt_area':(b[2]-b[0])*(b[3]-b[1])}
     for kind,idx in (('final',final),('oracle',oracle)):
      base[f'{kind}_exists']=idx is not None
      if idx is None:continue
      pb=box[idx];base.update({f'{kind}_iou':float(ov[idx]),f'{kind}_score':float(score[idx,c]),f'{kind}_level':str(level[idx]),f'{kind}_center_error':float(math.hypot((pb[0]+pb[2]-b[0]-b[2])/2,(pb[1]+pb[3]-b[1]-b[3])/2)),f'{kind}_width_rel_error':float((pb[2]-pb[0]-(b[2]-b[0]))/(b[2]-b[0])),f'{kind}_height_rel_error':float((pb[3]-pb[1]-(b[3]-b[1]))/(b[3]-b[1]))})
      ax=(anchors[idx,0]-pad[0])/gain[1];ay=(anchors[idx,1]-pad[1])/gain[0];true=np.array([ax-b[0],ay-b[1],b[2]-ax,b[3]-ay])*gain[0]/strides[idx]
      for ei,e in enumerate(EDGES):
       pp=prob[idx,ei];base[f'{kind}_{e}_entropy']=float(-(pp*np.log(pp+1e-12)).sum());base[f'{kind}_{e}_maxprob']=float(pp.max());base[f'{kind}_{e}_argmax']=int(pp.argmax());base[f'{kind}_{e}_variance']=float((pp*(np.arange(16)-exp[idx,ei])**2).sum());base[f'{kind}_{e}_decoded_distance']=float(exp[idx,ei]);base[f'{kind}_{e}_target_distance']=float(true[ei]);base[f'{kind}_{e}_distance_error']=float(exp[idx,ei]-true[ei])
     rows.append(base)
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with (a.output_dir/'dfl_instance_diagnosis.csv').open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
 # Stable aggregate correlations and error decomposition.
 summary={}
 for model in ('e0','cbcd'):
  q=[r for r in rows if r['model']==model];summary[model]={'instances':len(q)}
  for kind in ('final','oracle'):
   z=[r for r in q if r.get(f'{kind}_exists')];ent=np.array([np.mean([r[f'{kind}_{e}_entropy'] for e in EDGES]) for r in z]);iv=np.array([r[f'{kind}_iou'] for r in z]);summary[model][kind]={'exists':len(z),'mean_iou':float(iv.mean()),'iou75':int((iv>=.75).sum()),'mean_center_error_px':float(np.mean([r[f'{kind}_center_error'] for r in z])),'mean_abs_width_rel_error':float(np.mean([abs(r[f'{kind}_width_rel_error']) for r in z])),'mean_abs_height_rel_error':float(np.mean([abs(r[f'{kind}_height_rel_error']) for r in z])),'entropy_iou_spearman':float(spearmanr(ent,iv).statistic)}
 summary['oracle_advantage']={m:{'oracle_ge75_final_lt75':sum(r.get('oracle_iou',0)>=.75 and r.get('final_iou',0)<.75 for r in rows if r['model']==m),'oracle_ge50_final_lt50':sum(r.get('oracle_iou',0)>=.5 and r.get('final_iou',0)<.5 for r in rows if r['model']==m)} for m in ('e0','cbcd')}
 (a.output_dir/'dfl_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 # Visual summaries.
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for m,col in [('e0','tab:blue'),('cbcd','tab:orange')]:
  q=[r for r in rows if r['model']==m and r.get('final_exists')];axs[0].scatter([np.mean([r[f'final_{e}_entropy'] for e in EDGES]) for r in q],[r['final_iou'] for r in q],s=9,alpha=.5,label=m,c=col);axs[1].scatter([r['final_center_error'] for r in q],[r['final_iou'] for r in q],s=9,alpha=.5,label=m,c=col)
 axs[0].set(xlabel='Mean DFL entropy',ylabel='IoU');axs[1].set(xlabel='Center error (original px)',ylabel='IoU');axs[0].legend();fig.tight_layout();fig.savefig(a.output_dir/'dfl_entropy_geometry.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
