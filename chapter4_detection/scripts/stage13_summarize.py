#!/usr/bin/env python3
"""Summarize Stage13 dynamics and relate geometry to Stage12 OOF generation."""
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage13',type=Path,required=True);p.add_argument('--stage12',type=Path,required=True);a=p.parse_args();run=a.stage13/'runs/e0_fold0_seed42_assignment_diagnostic';dyn=list(csv.DictReader((run/'assignment_dynamics.csv').open()))
 numeric=('epoch','class_id','gt_width','gt_height','inside_count','positive_count','positive_iou_mean','positive_iou_max','target_score_sum','positive_p3','positive_p4','positive_p5')
 for r in dyn:
  for k in numeric:r[k]=float(r[k]) if r[k] not in ('',None) else None
 epochs=sorted(set(int(r['epoch']) for r in dyn));summary={}
 for e in epochs:
  q=[r for r in dyn if r['epoch']==e];small=[r for r in q if min(r['gt_width'],r['gt_height'])<8]
  def s(z):return {'instances':len(z),'zero_positive':sum(r['positive_count']==0 for r in z),'zero_effective_weight':sum(r['target_score_sum']<=1e-9 for r in z),'positive_count_mean':float(np.mean([r['positive_count'] for r in z])),'positive_iou_mean':float(np.mean([r['positive_iou_mean'] for r in z if r['positive_iou_mean'] is not None])),'target_score_mean':float(np.mean([r['target_score_sum'] for r in z])),'p3_fraction':float(sum(r['positive_p3'] for r in z)/max(1,sum(r['positive_count'] for r in z)))}
  qi=[r for r in q if r['positive_iou_mean'] is not None];summary[str(e)]={'all':s(q),'lt8':s(small),'spearman_min_side_positive_count':float(spearmanr([min(r['gt_width'],r['gt_height']) for r in q],[r['positive_count'] for r in q]).statistic),'spearman_positive_iou_target_score':float(spearmanr([r['positive_iou_mean'] for r in qi],[r['target_score_sum'] for r in qi]).statistic)}
 (a.stage13/'assignment_dynamics_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Join fixed geometric descriptors with Stage12 OOF outcomes; this is descriptive, not training supervision history.
 cov={(r['image'],r['gt_id']):r for r in csv.DictReader((a.stage13/'anchor_coverage_instances.csv').open())};fail=list(csv.DictReader((a.stage12/'gt_failure_decomposition.csv').open()));rows=[]
 for r in fail:
  z=cov[(r['image'],r['gt_id'])];rows.append({**r,'raw_inside_all':int(z['raw_inside_all']),'tal_inside_all':int(z['tal_inside_all']),'input_min_side':min(float(z['input_width']),float(z['input_height'])),'local_contrast':float(z['local_contrast'])})
 keys=list(rows[0]);
 with (a.stage13/'coverage_candidate_relation.csv').open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
 rel={}
 for m in ('e0','cbcd'):
  q=[r for r in rows if r['model']==m];rel[m]={}
  for label,sel in [('raw0',[r for r in q if r['raw_inside_all']==0]),('raw1',[r for r in q if r['raw_inside_all']==1]),('raw2plus',[r for r in q if r['raw_inside_all']>=2]),('lt8',[r for r in q if r['input_min_side']<8])]:rel[m][label]={'n':len(sel),'generation_failure':sum(r['category']=='A_generation_insufficient' for r in sel),'rate':float(np.mean([r['category']=='A_generation_insufficient' for r in sel])) if sel else None}
 (a.stage13/'assignment_candidate_relation_summary.json').write_text(json.dumps(rel,indent=2)+'\n')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].plot(epochs,[summary[str(e)]['all']['positive_count_mean'] for e in epochs],marker='o',label='all');axs[0].plot(epochs,[summary[str(e)]['lt8']['positive_count_mean'] for e in epochs],marker='o',label='augmented min-side<8');axs[0].set(xlabel='epoch',ylabel='mean positives/GT');axs[0].legend();axs[1].plot(epochs,[summary[str(e)]['all']['positive_iou_mean'] for e in epochs],marker='o',label='positive IoU');axs[1].plot(epochs,[summary[str(e)]['all']['target_score_mean'] for e in epochs],marker='o',label='target-score sum');axs[1].set(xlabel='epoch');axs[1].legend();fig.tight_layout();fig.savefig(a.stage13/'assignment_dynamics.png',dpi=180);plt.close(fig)
 print(json.dumps({'dynamics':summary,'relation':rel},indent=2))
if __name__=='__main__':main()
