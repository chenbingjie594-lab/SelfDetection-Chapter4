#!/usr/bin/env python3
"""Compare cached/fresh FSPR patch centers with dense full-map outputs."""
import argparse,json
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from chapter4_detection.models.fspr import FSPR
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--images',type=int,default=3);a=p.parse_args();d=torch.load(a.cache,map_location='cpu',weights_only=False);ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);side=FSPR(*ck['channels'],variant=ck['variant']);side.load_state_dict(ck['state_dict']);side.eval();model=YOLO(str(a.weights)).model.cpu().eval();cap={};hs=[model.model[2].register_forward_hook(lambda m,i,o:cap.__setitem__('p2',o.detach())),model.model[16].register_forward_hook(lambda m,i,o:cap.__setitem__('p3',o.detach()))];lb=LetterBox(new_shape=(1536,1536),auto=False,stride=32);rows=[]
 with torch.no_grad():
  for ii in range(a.images):
   im=cv2.imread(d['images'][ii]);arr=lb(image=im);x=torch.from_numpy(arr[:,:,::-1].copy()).permute(2,0,1).float()[None]/255;model(x);full=side(cap['p2'],cap['p3']);u3=F.interpolate(cap['p3'],size=cap['p2'].shape[-2:],mode='nearest');pf2=F.pad(cap['p2'],(4,4,4,4));pf3=F.pad(u3,(4,4,4,4));pm=F.pad(torch.ones(cap['p2'].shape[-2:],dtype=torch.bool),(4,4,4,4))
   for k in [j for j,q in enumerate(d['meta']) if q['image_index']==ii][:64]:
    q=d['meta'][k];sy,sx=q['patch_origin_y'],q['patch_origin_x'];ly,lx=q['local_y'],q['local_x'];fresh2=pf2[:,:,sy+4:sy+12,sx+4:sx+12];fresh3=pf3[:,:,sy+4:sy+12,sx+4:sx+12];mask=pm[sy+4:sy+12,sx+4:sx+12][None];fo=side(fresh2,fresh3,mask,origin=(0,0));co=side(d['p2'][k:k+1].float(),d['p3_up'][k:k+1].float(),d['valid_mask'][k:k+1],origin=(0,0));errs={}
    for name in ('logits','offset','edges'):
     truth=full[name][0,:,q['p2_y'],q['p2_x']];errs[f'fresh_{name}_max_abs']=float((fo[name][0,:,ly,lx]-truth).abs().max());errs[f'cache_{name}_max_abs']=float((co[name][0,:,ly,lx]-truth).abs().max())
    rows.append({'image':q['image'],'p2_x':q['p2_x'],'p2_y':q['p2_y'],**errs})
 for h in hs:h.remove()
 rep={'fold':d['fold'],'variant':ck['variant'],'positions':len(rows),'fresh_max_abs':max(max(r[k] for r in rows) for k in r if k.startswith('fresh_')) if False else max(max(v for k,v in r.items() if k.startswith('fresh_')) for r in rows),'cache_max_abs':max(max(v for k,v in r.items() if k.startswith('cache_')) for r in rows),'cache_mean_abs':float(np.mean([v for r in rows for k,v in r.items() if k.startswith('cache_')])), 'rows':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps({k:v for k,v in rep.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
