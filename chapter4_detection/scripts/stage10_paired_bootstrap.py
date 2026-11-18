#!/usr/bin/env python3
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from chapter4_detection.scripts.paired_group_bootstrap import image_stats,prepare_weighted,weighted_evaluate
METRICS=('map50_95','ap75','flash_ap50_95','black_ap50_95')
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--candidate-name',default='CBCD');p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=5000);p.add_argument('--seed',type=int,default=20261118);a=p.parse_args()
 rec=json.loads(a.manifest.read_text())['records'];groups=defaultdict(list)
 for r in rec:
  if r['split']=='train':groups[r['source_group']].append(r['image_name'])
 ig={n:g for g,ns in groups.items() for n in ns};prep={}
 for key,path in [('e0',a.baseline),('cbcd',a.candidate)]:
  d=json.loads(path.read_text())['images'];prep[key]=prepare_weighted({n:image_stats(v,a.labels/f'{Path(n).stem}.txt') for n,v in d.items()},ig)
 rng=np.random.default_rng(a.seed);gn=sorted(groups);vals=[]
 for _ in range(a.repeats):
  s=rng.choice(gn,len(gn),replace=True);mul={g:int(np.sum(s==g)) for g in set(s)};vals.append(weighted_evaluate(prep['cbcd'],mul)-weighted_evaluate(prep['e0'],mul))
 z=np.asarray(vals);out={'protocol':{'comparison':f'{a.candidate_name}-E0','unit':'source_group','groups':len(gn),'images':138,'repeats':a.repeats,'single_training_seed':42},'intervals':{m:{'mean':float(z[:,i].mean()),'ci95':[float(x) for x in np.quantile(z[:,i],[.025,.975])],'p_gt_0':float(np.mean(z[:,i]>0))} for i,m in enumerate(METRICS)}}
 a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
