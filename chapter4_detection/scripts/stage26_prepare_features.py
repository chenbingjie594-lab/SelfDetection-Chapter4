#!/usr/bin/env python3
"""Cache aligned local P2/P3 patches with real GT heatmap and box targets."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
def lp(im):return Path(str(im).replace('/images/','/labels/')).with_suffix('.txt')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--images-list',type=Path,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--negative-per-image',type=int,default=64);a=p.parse_args();rng=np.random.default_rng(a.seed+a.fold);a.output.parent.mkdir(parents=True,exist_ok=True);model=YOLO(str(a.weights)).model.cpu().eval();cap={};h2=model.model[2].register_forward_hook(lambda m,i,o:cap.__setitem__('p2',o.detach()));h3=model.model[16].register_forward_hook(lambda m,i,o:cap.__setitem__('p3',o.detach()));lb=LetterBox(new_shape=(1536,1536),auto=False,stride=32);ims=[Path(x) for x in a.images_list.read_text().splitlines()];p2s=[];p3s=[];masks=[];locs=[];heat=[];offs=[];edges=[];pos=[];meta=[];start=time.time()
 with torch.no_grad():
  for ii,path in enumerate(ims):
   im=cv2.imread(str(path));h0,w0=im.shape[:2];arr=lb(image=im);x=torch.from_numpy(arr[:,:,::-1].copy()).permute(2,0,1).float()[None]/255;out=model(x);f2=cap['p2'].float();f3=cap['p3'].float();u3=F.interpolate(f3,size=f2.shape[-2:],mode='nearest');_,c,H,W=f2.shape;raw=out[1];n3=raw['feats'][0].shape[-2]*raw['feats'][0].shape[-1];e0=raw['scores'][0,:,:n3].sigmoid().max(0).values.reshape(raw['feats'][0].shape[-2:]).repeat_interleave(2,0).repeat_interleave(2,1)[:H,:W]
   labs=[list(map(float,z.split())) for z in lp(path).read_text().splitlines() if z.strip()];gain=min(1536/w0,1536/h0);nw,nh=round(w0*gain),round(h0*gain);px=(1536-nw)/2;py=(1536-nh)/2;gt=[]
   for gid,q in enumerate(labs):
    cl,cx,cy,bw,bh=q;cx=(cx*w0*gain+px)/4;cy=(cy*h0*gain+py)/4;ww=bw*w0*gain/4;hh=bh*h0*gain/4;gt.append((gid,int(cl),cx,cy,ww,hh))
   samples={}
   for gid,cl,cx,cy,ww,hh in gt:
    gx,gy=int(np.floor(cx)),int(np.floor(cy))
    for yy in range(max(0,gy-1),min(H,gy+2)):
     for xx in range(max(0,gx-1),min(W,gx+2)):
      dist=(xx+.5-cx)**2+(yy+.5-cy)**2;old=samples.get((yy,xx));v=(dist,gid,cl,cx,cy,ww,hh)
      if old is None or dist<old[0]:samples[(yy,xx)]=v
   valid=torch.zeros(H,W,dtype=torch.bool);valid[int(py/4):int(np.ceil((py+nh)/4)),int(px/4):int(np.ceil((px+nw)/4))]=True;valid&=e0<.05
   yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
   for gid,cl,cx,cy,ww,hh in gt:valid&=~((xx>=cx-ww)&(xx<=cx+ww)&(yy>=cy-hh)&(yy<=cy+hh))
   for k in samples:valid[k]=False
   negidx=torch.where(valid.flatten())[0].numpy();negidx=rng.choice(negidx,min(len(negidx),a.negative_per_image),replace=False)
   items=[(y,x0,v) for (y,x0),v in samples.items()]+[(int(j//W),int(j%W),None) for j in negidx]
   pf2=F.pad(f2,(4,4,4,4));pf3=F.pad(u3,(4,4,4,4));pm=F.pad(torch.ones((H,W),dtype=torch.bool),(4,4,4,4))
   for y0,x0,v in items:
    sy=((y0-3)//2)*2;sx=((x0-3)//2)*2;ly=y0-sy;lx=x0-sx;p2s.append(pf2[0,:,sy+4:sy+12,sx+4:sx+12].half());p3s.append(pf3[0,:,sy+4:sy+12,sx+4:sx+12].half());masks.append(pm[sy+4:sy+12,sx+4:sx+12]);locs.append((ly,lx));ispos=v is not None;pos.append(ispos);ht=torch.zeros(2)
    if ispos:
     dist,gid,cl,cx,cy,ww,hh=v;ht[cl]=np.exp(-dist/2);offs.append((cx-(x0+.5),cy-(y0+.5)));edges.append((ww/2,hh/2,ww/2,hh/2))
    else:gid=cl=-1;offs.append((0,0));edges.append((0,0,0,0))
    heat.append(ht);meta.append({'image':path.name,'image_index':ii,'p2_x':x0,'p2_y':y0,'patch_origin_x':sx,'patch_origin_y':sy,'local_x':lx,'local_y':ly,'positive':ispos,'gt_id':gid,'class_id':cl,'e0_score':float(e0[y0,x0])})
   print(a.fold,ii+1,len(ims),path.name,'positive',len(samples),'negative',len(negidx),flush=True)
 h2.remove();h3.remove();d={'fold':a.fold,'images':[str(x) for x in ims],'weights':str(a.weights.resolve()),'weights_sha256':sha(a.weights),'p2':torch.stack(p2s),'p3_up':torch.stack(p3s),'valid_mask':torch.stack(masks),'local_index':torch.tensor(locs),'heat':torch.stack(heat),'offset':torch.tensor(offs),'edges':torch.tensor(edges),'positive':torch.tensor(pos),'meta':meta,'channels':[f2.shape[1],f3.shape[1]],'protocol':{'imgsz':1536,'stride':4,'patch':8,'patch_origin':'even aligned','positive_radius':1,'negative_per_image':a.negative_per_image,'negative_e0_score_max':.05,'gt_protection_expand':2.0}}
 torch.save(d,a.output);rep={'fold':a.fold,'images':len(ims),'samples':len(pos),'positive':sum(pos),'negative':len(pos)-sum(pos),'cache':str(a.output.resolve()),'sha256':sha(a.output),'bytes':a.output.stat().st_size,'seconds':time.time()-start};a.output.with_suffix('.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
