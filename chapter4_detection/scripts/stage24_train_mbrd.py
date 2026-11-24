#!/usr/bin/env python3
"""Train only the MBRD adapter through the frozen original P3 classification tower."""
import argparse,csv,hashlib,json,random,time,copy
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
from ultralytics import YOLO
from chapter4_detection.models.mbrd import MBRD
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--prototypes',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--mode',choices=('plain','fixed','matched'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=256);p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);d=torch.load(a.cache,map_location='cpu',weights_only=False);pd=torch.load(a.prototypes,map_location='cpu',weights_only=False);base=YOLO(str(a.weights)).model.cpu().eval();tower=copy.deepcopy(base.model[-1].cv3[0]).eval();tower.requires_grad_(False);m=MBRD(d['p3_shape'][0],pd['prototypes'],a.mode);ds=TensorDataset(d['patches'].float(),d['targets'].float(),d['positive']);dl=DataLoader(ds,batch_size=a.batch,shuffle=True,generator=torch.Generator().manual_seed(a.seed));opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);logs=[];start=time.time()
 with torch.no_grad():q=next(iter(dl));identity=float((m(q[0])-q[0]).abs().max())
 for ep in range(a.epochs):
  m.train();tot=n=gg=gf=0.
  for x,t,pos in dl:
   z=m(x);log=tower(z)[:,:,3,3];loss=F.binary_cross_entropy_with_logits(log,t);opt.zero_grad();loss.backward();gg+=float(m.gamma.grad.abs());gf+=sum(float(v.grad.norm()) for v in m.fuse.parameters() if v.grad is not None);opt.step();tot+=float(loss)*len(x);n+=len(x)
  logs.append({'epoch':ep+1,'loss':tot/n,'gamma':float(m.gamma.detach()),'gamma_grad_sum':gg,'fusion_grad_sum':gf});print(a.mode,ep+1,tot/n,float(m.gamma),flush=True)
 ck={'mode':a.mode,'state_dict':m.state_dict(),'config':{'k':32,'topk':4,'tau':.1,'hidden':32,'epochs':a.epochs,'batch':a.batch,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'seed':a.seed},'prototype_sha256':hashlib.sha256(a.prototypes.read_bytes()).hexdigest(),'e0_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'initial_identity_max_abs':identity};path=a.output/'last.pt';torch.save(ck,path)
 with (a.output/'training_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=logs[0].keys(),lineterminator='\n');w.writeheader();w.writerows(logs)
 with torch.no_grad():x,t,_=next(iter(dl));changed=float((tower(m(x))[:,:,3,3]-tower(x)[:,:,3,3]).abs().mean())
 rep={'mode':a.mode,'samples':len(ds),'positive':int(d['positive'].sum()),'background':int((~d['positive']).sum()),'parameters':sum(x.numel() for x in m.parameters()),'trainable_parameters':sum(x.numel() for x in m.parameters() if x.requires_grad),'initial_identity_max_abs':identity,'final_gamma':logs[-1]['gamma'],'final_loss':logs[-1]['loss'],'mean_abs_p3_logit_change':changed,'elapsed_seconds':time.time()-start,'checkpoint':str(path.resolve()),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint_bytes':path.stat().st_size};(a.output/'run.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
