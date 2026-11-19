#!/usr/bin/env python3
"""Attach inference-observable DFL statistics to traced candidates; no predictor is fitted."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=list(csv.DictReader(a.trace.open()));groups=defaultdict(list)
 for r in rows:groups[(r['model'],int(r['fold']),r['image'])].append(r)
 out=[]
 for (model,fold,image),q in groups.items():
  z=np.load(a.raw_root/f'{model}_fold{fold}'/f'{Path(image).stem}.npz');x=z['raw_boxes'].T.reshape(-1,4,16);anchors=sorted(set(int(r['anchor_id']) for r in q));xx=x[anchors];xx-=xx.max(2,keepdims=True);pr=np.exp(xx);pr/=pr.sum(2,keepdims=True);bins=np.arange(16);mean=(pr*bins).sum(2);var=(pr*(bins-mean[:,:,None])**2).sum(2);ent=-(pr*np.log(pr+1e-12)).sum(2);lookup={v:i for i,v in enumerate(anchors)}
  for r in q:
   i=lookup[int(r['anchor_id'])];out.append({**r,'dfl_entropy_mean':float(ent[i].mean()),'dfl_variance_mean':float(var[i].mean()),'dfl_maxprob_mean':float(pr[i].max(1).mean()),'distance_lr_asymmetry':float(abs(mean[i,0]-mean[i,2])),'distance_tb_asymmetry':float(abs(mean[i,1]-mean[i,3]))})
 keys=list(out[0]);
 with a.output.open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(out)
 summary={}
 for m in ('e0','cbcd'):
  top={}
  for r in out:
   if r['model']!=m:continue
   k=(r['image'],r['class_id']);
   if k not in top or float(r['score'])>float(top[k]['score']):top[k]=r
  z=list(top.values()); y=[float(r['quality_iou']) for r in z];summary[m]={k:float(spearmanr([float(r[k]) for r in z],y).statistic) for k in ('score','dfl_entropy_mean','dfl_variance_mean','dfl_maxprob_mean','distance_lr_asymmetry','distance_tb_asymmetry')}
 a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
