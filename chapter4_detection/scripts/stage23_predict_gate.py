#!/usr/bin/env python3
"""Apply fixed or learned QPR gate to frozen outer-fold CIR-C residuals."""
import argparse,json,time
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from chapter4_detection.models.cir import CIRRefiner,apply_residual
from chapter4_detection.models.qpr_cir import QPRGate,gate_features
from chapter4_detection.scripts.stage22_common import CIRCandidates
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--cir-checkpoint',type=Path,required=True);p.add_argument('--gate-checkpoint',type=Path);p.add_argument('--fixed-gate',type=float);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert (a.gate_checkpoint is None)!=(a.fixed_gate is None)
 cc=torch.load(a.cir_checkpoint,map_location='cpu',weights_only=False);cir=CIRRefiner('C');cir.load_state_dict(cc['state_dict']);cir.eval();gate=mode=None
 if a.gate_checkpoint:
  gc=torch.load(a.gate_checkpoint,map_location='cpu',weights_only=False);mode=gc['mode'];gate=QPRGate(mode);gate.load_state_dict(gc['state_dict']);gate.eval()
 ds=CIRCandidates(a.candidates);dl=DataLoader(ds,batch_size=32,shuffle=False);vals=[];start=time.time()
 with torch.no_grad():
  for q in dl:
   fh=cir.encoder(q['high']);fl=cir.encoder(q['low']);d=cir(q['high'],q['low'],q['semantic']);g=torch.full((len(d),),a.fixed_gate) if gate is None else gate(gate_features(q['semantic'],d,fh,fl,mode));b=apply_residual(q['box'],g[:,None]*d)
   vals += [(b[j].tolist(),d[j].tolist(),float(g[j])) for j in range(len(g))]
 src=json.loads(a.candidates.read_text());images=defaultdict(lambda:{'width':4096,'height':1024,'predictions':[]});trace=[]
 for c,(b,d,g) in zip(src['candidates'],vals):
  b=[max(0,min(4096,b[0])),max(0,min(1024,b[1])),max(0,min(4096,b[2])),max(0,min(1024,b[3]))];images[c['image']]['predictions'].append({'class_id':c['class_id'],'confidence':c['confidence'],'xyxy':b});trace.append({'candidate_id':c['candidate_id'],'image':c['image'],'class_id':c['class_id'],'confidence':c['confidence'],'original_xyxy':c['xyxy'],'cir_residual':d,'gate':g,'gated_xyxy':b})
 out={'schema_version':1,'model':f"QPR-{mode or a.fixed_gate}",'protocol':'frozen E0/CIR; gate changes geometry only','images_count':len(images),'images':dict(images),'elapsed_seconds':time.time()-start};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out)+'\n');a.output.with_name(a.output.stem+'_trace.json').write_text(json.dumps(trace)+'\n');print({'images':len(images),'candidates':len(trace),'seconds':out['elapsed_seconds']})
if __name__=='__main__':main()
