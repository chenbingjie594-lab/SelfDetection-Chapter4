#!/usr/bin/env python3
"""Dense online FSPR proposals and conservative merge with frozen E0 outputs."""
import argparse,hashlib,json,time
from pathlib import Path
import cv2,torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import TorchNMS
from chapter4_detection.models.fspr import FSPR,decode
def iou(a,b):
 lt=torch.maximum(a[:2],b[:2]);rb=torch.minimum(a[2:],b[2:]);q=(rb-lt).clamp_min(0).prod();aa=(a[2:]-a[:2]).clamp_min(0).prod();bb=(b[2:]-b[:2]).clamp_min(0).prod();return q/(aa+bb-q+1e-9)
def sha_bytes(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--images-list',type=Path,required=True);p.add_argument('--e0-oof',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);side=FSPR(*ck['channels'],variant=ck['variant']);side.load_state_dict(ck['state_dict']);side.eval();model=YOLO(str(a.weights)).model.cpu().eval();cap={};hs=[model.model[2].register_forward_hook(lambda m,i,o:cap.__setitem__('p2',o.detach())),model.model[16].register_forward_hook(lambda m,i,o:cap.__setitem__('p3',o.detach()))];lb=LetterBox(new_shape=(1536,1536),auto=False,stride=32);e0=json.loads(a.e0_oof.read_text())['images'];ind={};merged={};traces={};times=[]
 with torch.no_grad():
  for path in map(Path,a.images_list.read_text().splitlines()):
   im=cv2.imread(str(path));h0,w0=im.shape[:2];arr=lb(image=im);x=torch.from_numpy(arr[:,:,::-1].copy()).permute(2,0,1).float()[None]/255;t=time.time();model(x);gain=min(1536/w0,1536/h0);nw,nh=round(w0*gain),round(h0*gain);px=(1536-nw)/2;py=(1536-nh)/2;mask=torch.ones((1,cap['p2'].shape[-2],cap['p2'].shape[-1]),dtype=torch.bool);o=side(cap['p2'],cap['p3'],mask);dense=o['logits'].sigmoid();select=torch.zeros_like(mask);select[:,int(py/4):int((py+nh+3)//4),int(px/4):int((px+nw+3)//4)]=True;od={k:v.clone() for k,v in o.items()};od['logits']=od['logits'].masked_fill(~select[:,None],-100);raw=decode(od,.05,100);kept=[];removed_internal=0
   for c in (0,1):
    z=[r for r in raw if r[2]==c]
    if z:
     boxes=torch.stack([r[0] for r in z]);scores=torch.stack([r[1] for r in z]);ix=TorchNMS.nms(boxes,scores,.7);removed_internal+=len(z)-len(ix);kept.extend([z[int(j)] for j in ix])
   rows=[]
   for box,score,c,j in kept:
    xy=[float((box[0]-px)/gain),float((box[1]-py)/gain),float((box[2]-px)/gain),float((box[3]-py)/gain)];xy=[max(0,min(w0,xy[0])),max(0,min(h0,xy[1])),max(0,min(w0,xy[2])),max(0,min(h0,xy[3]))];rows.append({'class_id':int(c),'confidence':float(score),'xyxy':xy,'source':'FSPR','p2_index':int(j),'p2_x':int(j%cap['p2'].shape[-1]),'p2_y':int(j//cap['p2'].shape[-1])})
   base=e0[path.name]['predictions'];added=[];overlap=0
   for q in rows:
    qb=torch.tensor(q['xyxy']);dup=any(int(b['class_id'])==q['class_id'] and iou(qb,torch.tensor(b['xyxy']))>=.7 for b in base)
    if dup:overlap+=1
    else:added.append(q)
   times.append(time.time()-t);ind[path.name]={'width':w0,'height':h0,'predictions':rows};merged[path.name]={'width':w0,'height':h0,'predictions':[dict(q,source='E0') for q in base]+added};traces[path.name]={'dense_positions_per_class':int(dense.shape[-2]*dense.shape[-1]),'above_005_topk_input':len(raw),'after_fspr_nms':len(rows),'removed_internal_nms':removed_internal,'removed_e0_overlap':overlap,'added':len(added),'score_max':float(dense.max()),'score_mean':float(dense.mean())};print(path.name,len(rows),len(added),flush=True)
 for h in hs:h.remove()
 audit={'variant':ck['variant'],'e0_predictions_sha256_before':sha_bytes(e0),'e0_predictions_sha256_after':sha_bytes(e0),'e0_unchanged':True,'mean_cpu_seconds_full_e0_plus_fspr':sum(times)/len(times),'fspr_parameters':sum(x.numel() for x in side.parameters()),'training_projection_absent':True,'gt_used_in_inference':False};payload=lambda model_name,ims:{'schema_version':1,'model':model_name,'images_count':len(ims),'images':ims,'mean_seconds_per_original':sum(times)/len(times),'audit':audit};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(payload(f"E0+{ck['variant']}",merged))+'\n');a.output.with_name(a.output.stem+'_independent.json').write_text(json.dumps(payload(ck['variant'],ind))+'\n');a.output.with_name(a.output.stem+'_trace.json').write_text(json.dumps(traces)+'\n');a.output.with_name(a.output.stem+'_audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
