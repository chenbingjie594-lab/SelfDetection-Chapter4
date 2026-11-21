#!/usr/bin/env python3
"""Train one frozen Stage22 CIR ablation with no holdout validation."""
import argparse,csv,json,random,time,hashlib
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from chapter4_detection.models.cir import CIRRefiner,apply_residual
from chapter4_detection.scripts.stage22_common import CIRCandidates,box_iou_diag

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--variant',choices=('A','B','C','C-noDiff'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=32);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42);a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);a.output.mkdir(parents=True,exist_ok=True)
 ds=CIRCandidates(a.candidates);gen=torch.Generator().manual_seed(a.seed);dl=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=0,generator=gen);model=CIRRefiner(a.variant).to(a.device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);logs=[]
 initial_max=0.;start=time.time()
 with torch.no_grad():
  q=next(iter(dl));initial_max=float(model(q['high'].to(a.device),q['low'].to(a.device),q['semantic'].to(a.device)).abs().max())
 for ep in range(a.epochs):
  model.train();tot=posn=n=0.;grad=0.
  for q in dl:
   hi,lo,se,box,tgt,pos=[q[k].to(a.device) for k in ('high','low','semantic','box','target','positive')];d=model(hi,lo,se);pr=apply_residual(box,d);gt=apply_residual(box,tgt);res=F.smooth_l1_loss(d[pos],tgt[pos]) if pos.any() else d.sum()*0;ioul=(1-box_iou_diag(pr[pos],gt[pos])).mean() if pos.any() else d.sum()*0;neg=(d[~pos]**2).mean() if (~pos).any() else d.sum()*0;loss=res+ioul+.05*neg;opt.zero_grad();loss.backward();grad+=sum(float(x.grad.norm()) for x in model.parameters() if x.grad is not None);opt.step();tot+=float(loss)*len(pos);posn+=int(pos.sum());n+=len(pos)
  logs.append({'epoch':ep+1,'loss':tot/n,'positive_seen':posn,'gradient_norm_sum':grad});print(a.variant,ep+1,tot/n,flush=True)
 ck={'variant':a.variant,'state_dict':model.state_dict(),'config':{'roi':128,'epochs':a.epochs,'batch':a.batch,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'seed':a.seed},'initial_residual_max':initial_max};path=a.output/'last.pt';torch.save(ck,path)
 with (a.output/'training_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(logs[0]),lineterminator='\n');w.writeheader();w.writerows(logs)
 report={'variant':a.variant,'device':a.device,'samples':len(ds),'parameters':sum(x.numel() for x in model.parameters()),'initial_residual_max':initial_max,'final_loss':logs[-1]['loss'],'elapsed_seconds':time.time()-start,'checkpoint':str(path.resolve()),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint_bytes':path.stat().st_size};(a.output/'run.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
