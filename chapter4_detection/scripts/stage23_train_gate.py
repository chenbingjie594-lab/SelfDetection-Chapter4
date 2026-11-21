#!/usr/bin/env python3
"""Train one QPR quality gate from cross-fitted CIR targets."""
import argparse,csv,hashlib,json,random,time
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from chapter4_detection.models.cir import apply_residual
from chapter4_detection.models.qpr_cir import QPRGate
from chapter4_detection.scripts.stage23_common import torch_iou
class QualityData(Dataset):
 def __init__(self,path,mode):self.r=json.loads(Path(path).read_text())['records'];self.mode=mode
 def __len__(self):return len(self.r)
 def __getitem__(self,i):
  r=self.r[i];return {'x':torch.tensor(r[self.mode]),'box':torch.tensor(r['box']),'gt':torch.tensor(r['gt_box']),'res':torch.tensor(r['cir_residual']),'positive':torch.tensor(r['positive']),'label':torch.tensor(r['quality_label'])}
def main():
 p=argparse.ArgumentParser();p.add_argument('--targets',type=Path,required=True);p.add_argument('--mode',choices=('geo','full'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=32);p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 ds=QualityData(a.targets,a.mode);dl=DataLoader(ds,batch_size=a.batch,shuffle=True,generator=torch.Generator().manual_seed(a.seed));model=QPRGate(a.mode);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);logs=[];start=time.time()
 for ep in range(a.epochs):
  total=grad=0.;n=0
  for q in dl:
   g=model(q['x']);pos=q['positive'];pr=apply_residual(q['box'],g[:,None]*q['res']);new=torch_iou(pr[pos],q['gt'][pos]);old=torch_iou(q['box'][pos],q['gt'][pos]);bce=F.binary_cross_entropy(g[pos],q['label'][pos]) if pos.any() else g.sum()*0;loc=(1-new).mean() if pos.any() else g.sum()*0;protect=F.relu(old-new).mean() if pos.any() else g.sum()*0;bg=.1*(g[~pos]**2).mean() if (~pos).any() else g.sum()*0;loss=bce+.5*loc+protect+bg;opt.zero_grad();loss.backward();grad+=sum(float(x.grad.norm()) for x in model.parameters() if x.grad is not None);opt.step();total+=float(loss)*len(g);n+=len(g)
  logs.append({'epoch':ep+1,'loss':total/n,'gradient_norm_sum':grad});print(a.mode,ep+1,total/n,flush=True)
 path=a.output/'last.pt';torch.save({'mode':a.mode,'state_dict':model.state_dict(),'config':{'epochs':a.epochs,'batch':a.batch,'lr':.001,'weight_decay':.0001,'lambda_loc':.5,'lambda_protect':1.,'lambda_bg':.1,'seed':a.seed}},path)
 with (a.output/'training_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=logs[0].keys(),lineterminator='\n');w.writeheader();w.writerows(logs)
 rep={'mode':a.mode,'samples':len(ds),'parameters':sum(x.numel() for x in model.parameters()),'final_loss':logs[-1]['loss'],'elapsed_seconds':time.time()-start,'checkpoint':str(path.resolve()),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checkpoint_bytes':path.stat().st_size};(a.output/'run.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
