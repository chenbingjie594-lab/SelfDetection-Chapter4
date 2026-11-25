#!/usr/bin/env python3
"""Build fold-local positive/easy/hard P3 patch caches without holdout access."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

def label_path(im): return Path(str(im).replace('/images/','/labels/')).with_suffix('.txt')
def labels(im): return [list(map(float,x.split())) for x in label_path(im).read_text().splitlines() if x.strip()]
def tal(crit,raw,batch):
    pd,ps=raw['boxes'].permute(0,2,1).contiguous(),raw['scores'].permute(0,2,1).contiguous();ap,st=make_anchors(raw['feats'],crit.stride,.5);dtype=ps.dtype
    imgsz=torch.tensor(raw['feats'][0].shape[2:],dtype=dtype)*crit.stride[0];targets=torch.cat((batch['batch_idx'][:,None],batch['cls'][:,None],batch['bboxes']),1);targets=crit.preprocess(targets,1,imgsz[[1,0,1,0]]);gl,gb=targets.split((1,4),2);mask=gb.sum(2,keepdim=True).gt_(0);pb=crit.bbox_decode(ap,pd)
    _,tb,ts,fg,gi=crit.assigner(ps.detach().sigmoid(),(pb.detach()*st).type(gb.dtype),ap*st,gl,gb,mask)
    return ts,fg,gi,gb
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--images-list',type=Path,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--easy-per-image',type=int,default=64);p.add_argument('--hard-per-class-image',type=int,default=16);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed+a.fold)
    model=YOLO(str(a.weights)).model.cpu().eval();crit=v8DetectionLoss(model);lb=LetterBox(new_shape=(1536,1536),auto=False,scale_fill=False,scaleup=True,stride=32);ims=[Path(x) for x in a.images_list.read_text().splitlines()]
    patches=[];patch_masks=[];targets=[];types=[];classes=[];weights=[];meta=[];candidates=[];start=time.time()
    with torch.no_grad():
      for ii,path in enumerate(ims):
        im=cv2.imread(str(path));h0,w0=im.shape[:2];arr=lb(image=im);x=torch.from_numpy(arr[:,:,::-1].copy()).permute(2,0,1).float()[None]/255;out=model(x);raw=out[1] if isinstance(out,tuple) else out;decoded=out[0][0,:4].T;f=raw['feats'][0].float();_,c,h,w=f.shape;n=h*w;lab=labels(path)
        gain=min(1536/w0,1536/h0);nw,nh=round(w0*gain),round(h0*gain);px=(1536-nw)/2;py=(1536-nh)/2;bb=[]
        for q in lab:
          _,cx,cy,bw,bh=q;bb.append([(cx*w0*gain+px)/1536,(cy*h0*gain+py)/1536,bw*w0*gain/1536,bh*h0*gain/1536])
        batch={'batch_idx':torch.zeros(len(lab)),'cls':torch.tensor([q[0] for q in lab]),'bboxes':torch.tensor(bb,dtype=torch.float32).reshape(-1,4)};ts,fg,gi,gb=tal(crit,raw,batch);t3=ts[0,:n];pos=t3.sum(1)>0;log=raw['scores'][0,:,:n].T;prob=log.sigmoid();mx,pc=prob.max(1);yy,xx=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
        gtxy=[]
        for cx,cy,bw,bh in bb:gtxy.append([(cx-bw/2)*1536,(cy-bh/2)*1536,(cx+bw/2)*1536,(cy+bh/2)*1536])
        if gtxy:
          g=torch.tensor(gtxy);b=decoded[:n];lt=torch.maximum(b[:,None,:2],g[None,:,:2]);rb=torch.minimum(b[:,None,2:],g[None,:,2:]);inter=(rb-lt).clamp_min(0).prod(2);area=(b[:,2:]-b[:,:2]).clamp_min(0).prod(1)[:,None];ga=(g[:,2:]-g[:,:2]).clamp_min(0).prod(1)[None];maxiou=(inter/(area+ga-inter+1e-9)).max(1).values;bc=(b[:,:2]+b[:,2:])/2;gc=(g[:,:2]+g[:,2:])/2;mindist=torch.cdist(bc,gc).min(1).values
        else:maxiou=torch.zeros(n);mindist=torch.full((n,),float('inf'))
        valid=(xx>=px/8)&(xx<(px+nw)/8)&(yy>=py/8)&(yy<(py+nh)/8)
        for q in bb:
          cx,cy,bw,bh=q;x1=(cx-bw)*1536/8;y1=(cy-bh)*1536/8;x2=(cx+bw)*1536/8;y2=(cy+bh)*1536/8;valid&=~((xx>=x1)&(xx<=x2)&(yy>=y1)&(yy<=y2))
        valid=valid.flatten()&~pos&(maxiou<.1);easy=torch.where(valid&(mx<.05))[0].numpy();easy=rng.choice(easy,min(len(easy),a.easy_per_image),replace=False).tolist()
        hard=[]
        for cl in (0,1):
          ids=torch.where(valid&(pc==cl)&(mx>=.05)&(mx<.25))[0].numpy();take=min(len(ids),a.hard_per_class_image);hard.extend(rng.choice(ids,take,replace=False).tolist())
        # The frozen YOLO classification tower uses zero padding.  Zero-pad the
        # cached crop as well so boundary centers are identical to full-map inference.
        uncertain=torch.where(valid&(mx>=.25))[0].tolist();pad=F.pad(f,(3,3,3,3),mode='constant',value=0)[0];padmask=F.pad(torch.ones((h,w)),(3,3,3,3),mode='constant',value=0)
        # Cap TAL locations per GT to prevent one object dominating; choose strongest target scores.
        positives=[]
        for g in range(len(lab)):
          ids=torch.where(pos&(gi[0,:n]==g))[0];ids=ids[t3[ids].max(1).values.argsort(descending=True)[:8]];positives.extend(ids.tolist())
        for typ,ids,wt in ((0,positives,1.),(1,easy,1.),(2,hard,.25)):
          for j in ids:
            y,jx=divmod(j,w);patches.append(pad[:,y:y+7,jx:jx+7].half());patch_masks.append(padmask[y:y+7,jx:jx+7].bool());targets.append(t3[j].half());types.append(typ);cl=int(t3[j].argmax()) if typ==0 else (int(pc[j]) if typ==2 else -1);classes.append(cl);weights.append(wt);meta.append({'image':path.name,'image_index':ii,'p3_index':j,'x':jx,'y':y,'sample_type':typ,'target_class':cl,'e0_score':float(prob[j,cl]) if cl>=0 else float(mx[j]),'gt_id':int(gi[0,j]) if typ==0 else -1})
        for status,ids in (('B_WEAK_UNCONFIRMED',hard),('C_EXCLUDED_SUSPECT',uncertain)):
          for j in ids:
            y,jx=divmod(j,w);ox=((jx+.5)*8-px)/gain;oy=((y+.5)*8-py)/gain;db=decoded[j];ob=[float((db[0]-px)/gain),float((db[1]-py)/gain),float((db[2]-px)/gain),float((db[3]-py)/gain)];candidates.append({'fold':a.fold,'image':path.name,'image_path':str(path),'p3_index':j,'p3_x':jx,'p3_y':y,'original_x':float(ox),'original_y':float(oy),'decoded_x1':ob[0],'decoded_y1':ob[1],'decoded_x2':ob[2],'decoded_y2':ob[3],'predicted_class':int(pc[j]),'e0_score':float(mx[j]),'max_iou_gt':float(maxiou[j]),'nearest_gt_center_distance_input':float(mindist[j]),'inside_expanded_gt':False,'review_status':status,'weak_weight':.25 if status.startswith('B_') else 0.,'reason':'0.05<=score<0.25; decoded IoU<0.10; outside 2x GT protection' if status.startswith('B_') else 'score>=0.25; decoded IoU<0.10; possible missed defect; excluded'})
        print(a.fold,ii+1,len(ims),path.name,'pos/easy/hard/C',len(positives),len(easy),len(hard),len(uncertain),flush=True)
    cache={'fold':a.fold,'weights_path':str(a.weights.resolve()),'weights_sha256':sha(a.weights),'images':[str(x) for x in ims],'patches':torch.stack(patches),'patch_masks':torch.stack(patch_masks),'targets':torch.stack(targets),'sample_type':torch.tensor(types),'target_class':torch.tensor(classes),'sample_weight':torch.tensor(weights),'meta':meta,'hard_candidates':candidates,'p3_shape':[c,h,w],'protocol':{'imgsz':1536,'patch':7,'patch_padding':'zero plus post-adapter valid mask (matches frozen classification tower)','positive_per_gt_max':8,'easy_score':'<0.05','hard_B_score':'[0.05,0.25)','C_score':'>=0.25','gt_protection_expand':2.,'B_weight':.25,'human_confirmed_A':0}}
    torch.save(cache,a.output);counts={k:int((cache['sample_type']==v).sum()) for k,v in {'positive':0,'easy':1,'hard_B':2}.items()};rep={'fold':a.fold,'images':len(ims),**counts,'candidate_B':sum(x['review_status'].startswith('B_') for x in candidates),'candidate_C':sum(x['review_status'].startswith('C_') for x in candidates),'human_confirmed_A':0,'cache':str(a.output.resolve()),'cache_sha256':sha(a.output),'cache_bytes':a.output.stat().st_size,'elapsed_seconds':time.time()-start};a.output.with_suffix('.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
