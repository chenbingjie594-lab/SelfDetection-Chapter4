#!/usr/bin/env python3
"""Create plots, grouped uncertainty, and NMS diagnostics for Stage12."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from chapter4_detection.scripts.evaluate_original import load_gt
from chapter4_detection.scripts.stage12_candidate_analysis import box_iou

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage12',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
 rows=list(csv.DictReader((a.stage12/'candidate_trace.csv').open()));gt=list(csv.DictReader((a.stage12/'gt_failure_decomposition.csv').open()))
 for r in rows:
  for k in ('score','quality_iou'):r[k]=float(r[k])
  r['class_id']=int(r['class_id'])
 # Source-group bootstrap of image/class top candidates.
 sg={r['image_name']:r['source_group'] for r in json.loads(a.manifest.read_text())['records'] if r['split']=='train'}; rng=np.random.default_rng(20261119); boot={}
 for model in ('e0','cbcd'):
  q=[r for r in rows if r['model']==model];top={}
  for r in q:
   k=(r['image'],r['class_id']);
   if k not in top or r['score']>top[k]['score']:top[k]=r
  groups=defaultdict(list)
  for r in top.values():groups[sg[r['image']]].append(r)
  gn=list(groups);vals=[]
  for _ in range(2000):
   z=[r for g in rng.choice(gn,len(gn),replace=True) for r in groups[g]]
   vals.append(spearmanr([r['score'] for r in z],[r['quality_iou'] for r in z]).statistic)
  boot[model]={'spearman':float(spearmanr([r['score'] for r in top.values()],[r['quality_iou'] for r in top.values()]).statistic),'group_bootstrap_ci95':[float(x) for x in np.quantile(vals,[.025,.975])]}
 # Per-intervention false/duplicate/match diagnostics.
 diag={}
 for f in sorted((a.stage12/'interventions').glob('*_predictions.json')):
  d=json.loads(f.read_text());bg=dup=n=0;hit50=hit75=0
  for name,rec in d['images'].items():
   tc,tb=load_gt(a.labels/f'{Path(name).stem}.txt',rec['width'],rec['height']);tc=tc.numpy().astype(int);tb=tb.numpy();matched50=set();matched75=set()
   for pr in rec['predictions']:
    n+=1; idx=np.where(tc==int(pr['class_id']))[0]; ov=box_iou(np.array([pr['xyxy']]),tb[idx])[0] if len(idx) else np.array([])
    if not len(ov) or ov.max()<.1:bg+=1
    if len(ov) and ov.max()>=.5:
     g=int(idx[ov.argmax()]);dup+=g in matched50;matched50.add(g)
    if len(ov) and ov.max()>=.75:matched75.add(int(idx[ov.argmax()]))
   hit50+=len(matched50);hit75+=len(matched75)
  diag[f.stem]={'predictions':n,'per_image':n/138,'background_iou_lt_0.1':bg,'duplicate_iou50':dup,'gt_matched_iou50':hit50,'gt_matched_iou75':hit75}
 (a.stage12/'stage12_diagnostics.json').write_text(json.dumps({'correlation_group_bootstrap':boot,'interventions':diag},indent=2)+'\n')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for i,m in enumerate(('e0','cbcd')):
  q=[r for r in rows if r['model']==m];ax[i].scatter([r['score'] for r in q],[r['quality_iou'] for r in q],s=5,alpha=.2);ax[i].set(xlabel='classification score',ylabel='same-class GT IoU',title=m.upper())
 fig.tight_layout();fig.savefig(a.stage12/'score_iou_scatter.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,4));data=[];labs=[]
 for m in ('e0','cbcd'):
  data.append([min(int(r['oracle_score_rank']),20) for r in gt if r['model']==m and r['oracle_score_rank']]);labs.append(m.upper())
 ax.hist(data,bins=np.arange(.5,20.6),label=labs,alpha=.55);ax.set(xlabel='oracle candidate score rank (clipped at 20)',ylabel='GT count');ax.legend();fig.tight_layout();fig.savefig(a.stage12/'oracle_rank_histogram.png',dpi=180);plt.close(fig)
 print(json.dumps({'bootstrap':boot,'interventions':diag},indent=2))
if __name__=='__main__':main()
