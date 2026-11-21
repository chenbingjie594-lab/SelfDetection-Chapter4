#!/usr/bin/env python3
"""Refine frozen post-NMS E0 candidates without changing class or score."""
import argparse,json,time
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from chapter4_detection.models.cir import CIRRefiner,apply_residual
from chapter4_detection.scripts.stage22_common import CIRCandidates
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False);model=CIRRefiner(ck['variant']).to(a.device);model.load_state_dict(ck['state_dict']);model.eval();ds=CIRCandidates(a.candidates);dl=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0);ref=[];start=time.time()
 with torch.no_grad():
  for q in dl:
   d=model(q['high'].to(a.device),q['low'].to(a.device),q['semantic'].to(a.device));b=apply_residual(q['box'].to(a.device),d).cpu();
   for j in range(len(b)):ref.append((b[j].tolist(),d[j].cpu().tolist()))
 src=json.loads(a.candidates.read_text());images=defaultdict(lambda:{'width':4096,'height':1024,'predictions':[]});trace=[]
 for c,(b,d) in zip(src['candidates'],ref):
  b[0]=max(0,min(4096,b[0]));b[2]=max(0,min(4096,b[2]));b[1]=max(0,min(1024,b[1]));b[3]=max(0,min(1024,b[3]));images[c['image']]['predictions'].append({'class_id':c['class_id'],'confidence':c['confidence'],'xyxy':b});trace.append({'candidate_id':c['candidate_id'],'image':c['image'],'class_id':c['class_id'],'confidence':c['confidence'],'original_xyxy':c['xyxy'],'refined_xyxy':b,'residual':d})
 out={'schema_version':1,'model':f"CIR-{ck['variant']}",'protocol':'frozen E0 post-NMS candidates; unchanged class/score/order; no second NMS','images_count':len(images),'images':dict(images),'elapsed_seconds':time.time()-start};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out)+'\n');a.output.with_name(a.output.stem+'_trace.json').write_text(json.dumps(trace)+'\n');print({'images':len(images),'candidates':len(trace),'seconds':out['elapsed_seconds']})
if __name__=='__main__':main()
