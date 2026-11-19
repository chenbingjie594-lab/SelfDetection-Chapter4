#!/usr/bin/env python3
"""Source-group bootstrap and information/localization relation for Stage15."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from chapter4_detection.scripts.paired_group_bootstrap import image_stats,prepare_weighted,weighted_evaluate
METRICS=('map50_95','ap75','flash_ap50_95','black_ap50_95')
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--b0',type=Path,required=True);p.add_argument('--b1',type=Path,required=True);p.add_argument('--b2',type=Path,required=True);p.add_argument('--candidate-csv',type=Path,required=True);p.add_argument('--information-csv',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=5000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rec=json.loads(a.manifest.read_text())['records'];groups=defaultdict(list)
 for r in rec:
  if r['split']=='train':groups[r['source_group']].append(r['image_name'])
 ig={n:g for g,ns in groups.items() for n in ns};prep={}
 for key,path in [('B0',a.b0),('B1',a.b1),('B2',a.b2)]:
  data=json.loads(path.read_text())['images'];prep[key]=prepare_weighted({n:image_stats(v,a.labels/f'{Path(n).stem}.txt') for n,v in data.items()},ig)
 rng=np.random.default_rng(20261119);gn=sorted(groups);vals={k:[] for k in ['B1-B0','B1-B2']}
 for _ in range(a.repeats):
  sampled=rng.choice(gn,len(gn),replace=True);mul={g:int(np.sum(sampled==g)) for g in set(sampled)}
  m={k:weighted_evaluate(v,mul) for k,v in prep.items()};vals['B1-B0'].append(m['B1']-m['B0']);vals['B1-B2'].append(m['B1']-m['B2'])
 boot={'protocol':{'unit':'source_group','groups':len(gn),'images':138,'repeats':a.repeats,'seed':20261119,'single_training_seed':42},'comparisons':{}}
 for comp,v in vals.items():
  z=np.asarray(v);boot['comparisons'][comp]={m:{'mean':float(z[:,i].mean()),'ci95':[float(x) for x in np.quantile(z[:,i],[.025,.975])],'p_gt_0':float(np.mean(z[:,i]>0))} for i,m in enumerate(METRICS)}
 (a.output/'source_group_bootstrap.json').write_text(json.dumps(boot,indent=2)+'\n')
 cand=pd.read_csv(a.candidate_csv);wide=cand.pivot(index=['image','gt_id','source_group','class_id','tiny'],columns='condition',values='qmax').reset_index();info=pd.read_csv(a.information_csv).rename(columns={'gt_index':'gt_id'});z=wide.merge(info,on=['image','gt_id','source_group','class_id']);z['delta_qmax_b1_b2']=z.B1-z.B2;z['delta_qmax_b1_b0']=z.B1-z.B0;z.to_csv(a.output/'information_candidate_relation.csv',index=False)
 metrics=['delta_contrast_gradient_mean','delta_contrast_laplacian_abs_mean','defect_psnr','defect_ssim'];relation={}
 for m in metrics:
  rho,pv=spearmanr(z[m],z.delta_qmax_b1_b2);relation[m]={'spearman_with_B1_minus_B2_qmax':float(rho),'p_value_descriptive':float(pv)}
 relation['mean_delta_qmax_b1_b2']=float(z.delta_qmax_b1_b2.mean());relation['tiny_mean_delta_qmax_b1_b2']=float(z.loc[z.tiny,'delta_qmax_b1_b2'].mean());relation['interpretation']='Descriptive association only; OOF was used in prior method development.'
 (a.output/'information_localization_relation.json').write_text(json.dumps(relation,indent=2)+'\n');print(json.dumps({'bootstrap':boot,'relation':relation},indent=2))
if __name__=='__main__':main()
