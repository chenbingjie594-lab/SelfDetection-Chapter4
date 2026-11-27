#!/usr/bin/env python3
"""Train one frozen-feature FSPR sidecar with GT heatmap and geometry supervision."""
import argparse,csv,hashlib,json,random,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from chapter4_detection.models.fspr import FSPR
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def iou_loss(off,ed,to,te):
 p=torch.stack((off[:,0]-ed[:,0],off[:,1]-ed[:,1],off[:,0]+ed[:,2],off[:,1]+ed[:,3]),1);g=torch.stack((to[:,0]-te[:,0],to[:,1]-te[:,1],to[:,0]+te[:,2],to[:,1]+te[:,3]),1);lt=torch.maximum(p[:,:2],g[:,:2]);rb=torch.minimum(p[:,2:],g[:,2:]);inter=(rb-lt).clamp_min(0).prod(1);a=(p[:,2:]-p[:,:2]).clamp_min(0).prod(1);b=(g[:,2:]-g[:,:2]).clamp_min(0).prod(1);return (1-inter/(a+b-inter+1e-8)).mean()
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--variant',choices=('F0','F1','F2'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=256);p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);d=torch.load(a.cache,map_location='cpu',weights_only=False);m=FSPR(*d['channels'],variant=a.variant);pos=d['positive'];sw=torch.where(pos,1/pos.sum(),1/(~pos).sum());sampler=WeightedRandomSampler(sw,len(pos),replacement=True,generator=torch.Generator().manual_seed(a.seed));ds=TensorDataset(d['p2'].float(),d['p3_up'].float(),d['valid_mask'],d['local_index'],d['heat'],d['offset'].float(),d['edges'].float(),pos);dl=DataLoader(ds,batch_size=a.batch,sampler=sampler);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);logs=[];start=time.time()
 for ep in range(a.epochs):
  m.train();sums=np.zeros(5);n=0;gr={k:0. for k in ('p2','p3','cls','off','edge')}
  for f2,f3,mask,loc,heat,to,te,positive in dl:
   o=m(f2,f3,mask,origin=(0,0));ix=torch.arange(len(f2));y,x=loc[:,0],loc[:,1];log=o['logits'][ix,:,y,x];off=o['offset'][ix,:,y,x];ed=o['edges'][ix,:,y,x];prob=log.sigmoid();bce=F.binary_cross_entropy_with_logits(log,heat,reduction='none');focal=torch.where(heat>0,(1-prob).pow(2),prob.pow(2))*bce;lc=focal.mean();pr=positive.bool()
   if pr.any():li=iou_loss(off[pr],ed[pr],to[pr],te[pr]);lo=F.smooth_l1_loss(off[pr],to[pr]);le=F.smooth_l1_loss(ed[pr],te[pr])
   else:li=lo=le=off.sum()*0
   loss=lc+li+lo+.5*le
   if not torch.isfinite(loss):raise RuntimeError('non-finite loss')
   opt.zero_grad();loss.backward()
   for k in gr:gr[k]+=sum(float(v.grad.norm()) for name,v in m.named_parameters() if name.startswith(k) and v.grad is not None)
   opt.step();sums+=np.array([float(loss),float(lc),float(li),float(lo),float(le)])*len(f2);n+=len(f2)
  row={'epoch':ep+1,'loss':sums[0]/n,'cls':sums[1]/n,'iou':sums[2]/n,'offset':sums[3]/n,'edge':sums[4]/n,**{f'{k}_grad':v for k,v in gr.items()}};logs.append(row);print(a.variant,*row.values(),flush=True)
 ck={'variant':a.variant,'state_dict':m.state_dict(),'channels':d['channels'],'config':{'width':24,'epochs':a.epochs,'batch':a.batch,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'seed':a.seed,'loss':'focal + IoU + offset + 0.5 edge','checkpoint':'last.pt','score_min':.05,'topk_per_class':100,'internal_nms':.7,'cross_source_iou':.7},'cache_sha256':sha(a.cache)};path=a.output/'last.pt';torch.save(ck,path)
 with (a.output/'training_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=logs[0],lineterminator='\n');w.writeheader();w.writerows(logs)
 rep={'variant':a.variant,'samples':len(ds),'positive':int(pos.sum()),'negative':int((~pos).sum()),'parameters':sum(x.numel() for x in m.parameters()),'final':logs[-1],'seconds':time.time()-start,'checkpoint':str(path.resolve()),'checkpoint_sha256':sha(path)};(a.output/'run.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
