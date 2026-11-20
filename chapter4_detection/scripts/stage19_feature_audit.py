#!/usr/bin/env python3
"""640/1536 train-only RGLCD audit with ordinary and hard backgrounds."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from chapter4_detection.models.rglcd import RGLCD,replace_detect_with_rglcd
from chapter4_detection.scripts.stage7_initialization_audit import real_batch
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--init',type=Path,required=True);ap.add_argument('--trained-e0',type=Path,required=True);ap.add_argument('--train-list',type=Path,required=True);ap.add_argument('--gt-stats',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 gtmeta={(r['image'],int(r['gt_id'])):r for r in csv.DictReader(a.gt_stats.open()) if r['split']=='train'}; init_state=YOLO(str(a.init)).model.model[-1].rglcd.state_dict();rows=[]
 for state,wp in [('INIT',a.init),('TRAINED_E0',a.trained_e0)]:
  model=YOLO(str(wp)).model.cpu().eval()
  if not isinstance(model.model[-1].rglcd if hasattr(model.model[-1],'rglcd') else None,RGLCD):replace_detect_with_rglcd(model);model.model[-1].rglcd.load_state_dict(init_state)
  head=model.model[-1];mod=head.rglcd;cap={};hook=head.register_forward_pre_hook(lambda m,x:cap.update(p3=x[0][0].detach()))
  for imgsz in (640,1536):
   for ip in a.train_list.read_text().splitlines():
    image=Path(ip);label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt');batch=real_batch(image,label,imgsz,'cpu')
    with torch.no_grad():model(batch['img']);p3=cap['p3'];mean,var,d,r,count=mod.statistics(p3);phi=mod.phi(d);gated=r*phi;score=head.cv3[0](p3).sigmoid().amax(1)[0]
    H,W=p3.shape[-2:]; boxes=batch['bboxes'];classes=batch['cls'].flatten().long();occupied=[]
    for j,b in enumerate(boxes):
     cx,cy,bw,bh=map(float,b);occupied.append((cx,cy,bw,bh));x=min(W-1,max(0,int(cx*W)));y=min(H-1,max(0,int(cy*H)));meta=gtmeta.get((image.name,j),{});kind='LOW_CONTRAST_BLACK' if int(classes[j])==1 and meta and float(meta['local_contrast'])<=.5 else ('FLASH_GT' if int(classes[j])==0 else 'BLACK_GT')
     add(rows,state,imgsz,image.name,j,kind,x,y,W,H,p3,d,var,r,phi,gated,score,count,meta,bw,bh)
    # Original images are 4:1 and real_batch uses centered square letterbox.
    # Exclude top/bottom padding from every background candidate.
    valid=torch.zeros(H,W,dtype=torch.bool);content_y1=int(round(H*3/8));content_y2=int(round(H*5/8));valid[content_y1:content_y2,:]=True
    for cx,cy,bw,bh in occupied:
     x1=max(0,int((cx-bw/2)*W)-3);x2=min(W,int((cx+bw/2)*W)+4);y1=max(0,int((cy-bh/2)*H)-3);y2=min(H,int((cy+bh/2)*H)+4);valid[y1:y2,x1:x2]=False
    # G1 fixed positions.
    for gy,gx in ((.42,.2),(.42,.8),(.58,.2),(.58,.8)):
     y,x=int(gy*(H-1)),int(gx*(W-1))
     if valid[y,x]:add(rows,state,imgsz,image.name,-1,'G1_ORDINARY',x,y,W,H,p3,d,var,r,phi,gated,score,count,{},0,0)
    contrast=d[0].abs().mean(0).masked_fill(~valid,-1);cls=score.masked_fill(~valid,-1)
    for kind,mapv in [('G2_HIGH_CONTRAST',contrast),('G3_HIGH_CLASS',cls)]:
     for idx in torch.topk(mapv.flatten(),k=min(4,int(valid.sum()))).indices.tolist():
      y,x=divmod(idx,W);add(rows,state,imgsz,image.name,-1,kind,x,y,W,H,p3,d,var,r,phi,gated,score,count,{},0,0)
   print(state,imgsz,'done')
  hook.remove()
 fields=list(rows[0]);
 with (a.output/'stage19_feature_instances.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 summary={}
 for state in ('INIT','TRAINED_E0'):
  for z in (640,1536):
   for kind in sorted({q['kind'] for q in rows}):
    rr=[q for q in rows if q['state']==state and q['imgsz']==z and q['kind']==kind]
    if rr:summary[f'{state}_{z}_{kind}']=summ(rr)
 (a.output/'stage19_feature_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
def add(rows,state,imgsz,image,gid,kind,x,y,W,H,p3,d,var,r,phi,gated,score,count,meta,bw,bh):
 u=float(torch.linalg.vector_norm(phi[0,:,y,x]));g=float(torch.linalg.vector_norm(gated[0,:,y,x]));rv=r[0,:,y,x]
 y1,y2=max(0,y-3),min(H,y+4);x1,x2=max(0,x-3),min(W,x+4);q=p3[0,:,y1:y2,x1:x2].float();cy=y-y1;cx=x-x1;dirs=[]
 if cy-1>0:dirs.append(q[:,:cy-1,:].mean((1,2)))
 if cy+2<q.shape[1]:dirs.append(q[:,cy+2:,:].mean((1,2)))
 if cx-1>0:dirs.append(q[:,:,:cx-1].mean((1,2)))
 if cx+2<q.shape[2]:dirs.append(q[:,:,cx+2:].mean((1,2)))
 directional=float(torch.stack(dirs).std(0,unbiased=False).mean()) if len(dirs)>1 else 0.0
 structure=float((q[:,:,1:]-q[:,:,:-1]).abs().mean()+(q[:,1:,:]-q[:,:-1,:]).abs().mean())/2
 rows.append({'state':state,'imgsz':imgsz,'image':image,'gt_id':gid,'kind':kind,'class_id':meta.get('class_id',''),'original_width':meta.get('width',''),'original_height':meta.get('height',''),'input_width':meta.get('input_width',''),'input_height':meta.get('input_height',''),'local_contrast':meta.get('local_contrast',''),'p3_h':H,'p3_w':W,'p3_x':x,'p3_y':y,'p3_box_w':bw*W,'p3_box_h':bh*H,'d_abs':float(d[0,:,y,x].abs().mean()),'variance_mean':float(var[0,:,y,x].mean()),'directional_inconsistency':directional,'structure_inconsistency':structure,'r_mean':float(rv.mean()),'r_min_channel':float(rv.min()),'r_max_channel':float(rv.max()),'phi_l2':u,'gated_l2':g,'suppression':1-g/(u+1e-8),'class_response':float(score[y,x]),'valid_count':float(count[0,0,y,x]),'boundary_affected':x<3 or y<3 or x>=W-3 or y>=H-3})
def summ(rr):
 out={'n':len(rr)}
 for k in ('d_abs','variance_mean','directional_inconsistency','structure_inconsistency','r_mean','phi_l2','gated_l2','suppression','class_response'):
  v=np.array([float(q[k]) for q in rr]);out[k]={'mean':float(v.mean()),'median':float(np.median(v)),'q05':float(np.quantile(v,.05)),'q25':float(np.quantile(v,.25)),'q75':float(np.quantile(v,.75)),'q95':float(np.quantile(v,.95)),'min':float(v.min()),'max':float(v.max())}
 return out
if __name__=='__main__':main()
