#!/usr/bin/env python3
"""Audit cached 7x7/FP16 center logits against full-map P3 inference."""
import argparse,json,copy
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from chapter4_detection.models.mbrd import MBRD
def tower_masked(tower,z,mask):
 z=tower[0](z)*mask[:,None];z=tower[1](z)*mask[:,None];return tower[2](z)
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--prototypes',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--images',type=int,default=3);a=p.parse_args();d=torch.load(a.cache,map_location='cpu',weights_only=False);pd=torch.load(a.prototypes,map_location='cpu',weights_only=False);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);base=YOLO(str(a.weights)).model.cpu().eval();tower=copy.deepcopy(base.model[-1].cv3[0]).eval().requires_grad_(False);m=MBRD(d['p3_shape'][0],pd['prototypes'],ck['mode']);m.load_state_dict(ck['state_dict']);m.eval();lb=LetterBox(new_shape=(1536,1536),auto=False,stride=32);errs=[];chosen=sorted(set(x['image_index'] for x in d['meta']))[:a.images]
 with torch.no_grad():
  for ii in chosen:
   path=Path(d['images'][ii]);im=cv2.imread(str(path));arr=lb(image=im);x=torch.from_numpy(arr[:,:,::-1].copy()).permute(2,0,1).float()[None]/255;out=base(x);raw=out[1] if isinstance(out,tuple) else out;f=raw['feats'][0].float();full=tower(m(f))[0];pad=F.pad(f,(3,3,3,3),mode='constant',value=0)[0]
   ids=[k for k,q in enumerate(d['meta']) if q['image_index']==ii][:64]
   for k in ids:
    q=d['meta'][k];y,xp=q['y'],q['x'];fresh=pad[:,y:y+7,xp:xp+7][None];cached=d['patches'][k:k+1].float();mask=d['patch_masks'][k:k+1];lf=tower_masked(tower,m(fresh)*mask[:,None],mask)[0,:,3,3];lc=tower_masked(tower,m(cached)*mask[:,None],mask)[0,:,3,3];truth=full[:,y,xp];errs.append({'image':q['image'],'p3_index':q['p3_index'],'fresh_patch_fullmap_max_abs':float((lf-truth).abs().max()),'fp16_cache_fullmap_max_abs':float((lc-truth).abs().max()),'fp16_cache_fullmap_mean_abs':float((lc-truth).abs().mean())})
 vals=lambda key:[z[key] for z in errs];rep={'fold':d['fold'],'images_checked':len(chosen),'positions_checked':len(errs),'bn_mode':'eval','patch_padding':'zero; matches full classification tower at P3 boundaries','fresh_patch_fullmap_max_abs':max(vals('fresh_patch_fullmap_max_abs')),'fp16_cache_fullmap_max_abs':max(vals('fp16_cache_fullmap_max_abs')),'fp16_cache_fullmap_mean_abs':float(np.mean(vals('fp16_cache_fullmap_mean_abs'))),'rows':errs};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({k:v for k,v in rep.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
