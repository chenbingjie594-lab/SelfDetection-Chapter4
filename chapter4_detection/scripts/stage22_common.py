"""Shared CIR crop, dataset, geometry and evaluation utilities."""
from __future__ import annotations
import json,math
from collections import OrderedDict
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import Dataset

def low_reconstruction(im):
 h,w=im.shape[:2];gain=min(1536/w,1536/h);small=cv2.resize(im,(round(w*gain),round(h*gain)),interpolation=cv2.INTER_LINEAR);return cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR)
def roi(box):
 x1,y1,x2,y2=map(float,box);cx=(x1+x2)/2;cy=(y1+y2)/2;side=min(256.,max(96.,4*max(x2-x1,y2-y1)));return np.array([cx-side/2,cy-side/2,cx+side/2,cy+side/2]),side
def crop(im,r,size=128):
 x1,y1,x2,y2=r;side=int(round(x2-x1));cx=(x1+x2)/2;cy=(y1+y2)/2
 # OpenCV getRectSubPix has no borderType: explicit replicate padding preserves both routes identically.
 pad=side+2;z=cv2.copyMakeBorder(im,pad,pad,pad,pad,cv2.BORDER_REPLICATE);patch=cv2.getRectSubPix(z,(side,side),(float(cx+pad),float(cy+pad)))
 return cv2.resize(patch,(size,size),interpolation=cv2.INTER_LINEAR)
def semantic(c,w=4096,h=1024):
 b=np.asarray(c['xyxy'],float);bw=b[2]-b[0];bh=b[3]-b[1];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
 return np.array([math.log(max(bw,1)/96),math.log(max(bh,1)/96),math.log(max(bw/bh,1e-6)),float(c['confidence']),float(c['class_id']),cx/w,cy/h,1.0],np.float32)
def target_residual(box,gt):
 b=np.asarray(box,float);g=np.asarray(gt,float);bw=b[2]-b[0];bh=b[3]-b[1];bcx=(b[0]+b[2])/2;bcy=(b[1]+b[3])/2;gcx=(g[0]+g[2])/2;gcy=(g[1]+g[3])/2
 return np.array([(gcx-bcx)/bw,(gcy-bcy)/bh,math.log((g[2]-g[0])/bw),math.log((g[3]-g[1])/bh)],np.float32)
class CIRCandidates(Dataset):
 def __init__(self,path,augment=False,seed=42):self.data=json.loads(Path(path).read_text())['candidates'];self.augment=augment;self.seed=seed;self.cache=OrderedDict();self.patch_cache={}
 def __len__(self):return len(self.data)
 def __getitem__(self,i):
  c=self.data[i];p=c['image_path']
  if i in self.patch_cache:xh,xl=self.patch_cache[i]
  else:
   xh=xl=None
  if xh is None and p not in self.cache:
   im=cv2.imread(p);self.cache[p]=(im,low_reconstruction(im));self.cache.move_to_end(p)
   if len(self.cache)>2:self.cache.popitem(last=False)
  elif xh is None:self.cache.move_to_end(p)
  if xh is None:
   hi,lo=self.cache[p];r,_=roi(c['xyxy']);xh=crop(hi,r);xl=crop(lo,r);self.patch_cache[i]=(xh,xl);shape=hi.shape
  else:shape=(1024,4096,3)
  to=lambda x:torch.from_numpy(x[:,:,::-1].copy()).permute(2,0,1).float()/255
  return {'high':to(xh),'low':to(xl),'semantic':torch.from_numpy(semantic(c,shape[1],shape[0])),'box':torch.tensor(c['xyxy'],dtype=torch.float32),'target':torch.tensor(c.get('target',[0,0,0,0]),dtype=torch.float32),'positive':torch.tensor(c['positive'],dtype=torch.bool),'image':c['image'],'candidate_id':c['candidate_id']}
def box_iou_diag(a,b):
 lt=torch.maximum(a[:,:2],b[:,:2]);rb=torch.minimum(a[:,2:],b[:,2:]);inter=(rb-lt).clamp_min(0).prod(1);aa=(a[:,2:]-a[:,:2]).clamp_min(0).prod(1);bb=(b[:,2:]-b[:,:2]).clamp_min(0).prod(1);return inter/(aa+bb-inter+1e-9)
