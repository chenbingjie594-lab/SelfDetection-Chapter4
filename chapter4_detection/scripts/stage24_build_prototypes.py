#!/usr/bin/env python3
"""Build deterministic fold-specific normal-background prototypes."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np,torch
from sklearn.cluster import KMeans
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--k',type=int,default=32);p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);d=torch.load(a.cache,map_location='cpu',weights_only=False);x=torch.nn.functional.normalize(d['background_vectors'].float(),dim=1).numpy();km=KMeans(n_clusters=a.k,random_state=a.seed,n_init=10,max_iter=300).fit(x);pvec=torch.nn.functional.normalize(torch.from_numpy(km.cluster_centers_).float(),dim=1);counts=np.bincount(km.labels_,minlength=a.k);members=[]
 for k in range(a.k):
  ids=np.where(km.labels_==k)[0];ims=Counter(d['background_meta'][i]['image'] for i in ids);members.append({'prototype':k,'samples':int(len(ids)),'images':len(ims),'dominant_image_fraction':max(ims.values())/len(ids),'texture_norm_mean':float(np.mean([d['background_meta'][i]['texture_norm'] for i in ids]))})
 sim=pvec@pvec.T;off=sim[~torch.eye(a.k,dtype=bool)];out={'fold':d['fold'],'k':a.k,'channels':pvec.shape[1],'prototypes':pvec,'members':members,'source_cache':str(a.cache.resolve()),'source_cache_sha256':hashlib.sha256(a.cache.read_bytes()).hexdigest(),'seed':a.seed};torch.save(out,a.output);rep={'fold':d['fold'],'k':a.k,'samples':len(x),'min_cluster':int(counts.min()),'max_cluster':int(counts.max()),'mean_images_per_prototype':float(np.mean([z['images'] for z in members])),'max_dominant_image_fraction':float(max(z['dominant_image_fraction'] for z in members)),'prototype_cosine_mean':float(off.mean()),'prototype_cosine_max':float(off.max()),'prototype_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'prototype_bytes':a.output.stat().st_size,'conservative_candidate_background':True,'expert_confirmed':False};a.output.with_suffix('.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
