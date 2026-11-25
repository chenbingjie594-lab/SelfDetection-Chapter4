#!/usr/bin/env python3
"""Train H1-H4 on one fold with identical balanced cached sampling."""
import argparse,csv,hashlib,json,random,time,copy
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from ultralytics import YOLO
from chapter4_detection.models.mbrd import MBRD
from chapter4_detection.models.hbd import HBDProjection,hbd_loss
CFG={'H1':('matched','none'),'H2':('fixed','hbd'),'H3':('matched','hbd'),'H4':('matched','generic')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tower_masked(tower,z,mask):
 z=tower[0](z)*mask[:,None];z=tower[1](z)*mask[:,None];return tower[2](z)
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--prototypes',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--variant',choices=CFG,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch',type=int,default=256);p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 d=torch.load(a.cache,map_location='cpu',weights_only=False);pd=torch.load(a.prototypes,map_location='cpu',weights_only=False);base=YOLO(str(a.weights)).model.cpu().eval();tower=copy.deepcopy(base.model[-1].cv3[0]).eval().requires_grad_(False);mode,kind=CFG[a.variant];m=MBRD(d['p3_shape'][0],pd['prototypes'],mode);proj=HBDProjection(d['p3_shape'][0],32)
 typ=d['sample_type'];counts=torch.bincount(typ,minlength=3).float();sw=(1/counts.clamp_min(1))[typ];sampler=WeightedRandomSampler(sw,len(typ),replacement=True,generator=torch.Generator().manual_seed(a.seed));ds=TensorDataset(d['patches'].float(),d['patch_masks'],d['targets'].float(),typ,d['target_class'],d['sample_weight']);dl=DataLoader(ds,batch_size=a.batch,sampler=sampler)
 params=list(m.parameters())+(list(proj.parameters()) if kind!='none' else []);opt=torch.optim.AdamW(params,lr=.001,weight_decay=.0001);logs=[];start=time.time();nan=False
 with torch.no_grad():q=next(iter(dl));identity=float((m(q[0])-q[0]).abs().max())
 for ep in range(a.epochs):
  m.train();proj.train();tot=clt=ct=0.;n=valid_batches=0;gm=gp=0.
  for x,mask,t,st,tc,wt in dl:
   z=m(x)*mask[:,None];log=tower_masked(tower,z,mask)[:,:,3,3];each=F.binary_cross_entropy_with_logits(log,t,reduction='none').mean(1);cls=(each*wt).sum()/wt.sum().clamp_min(1)
   contrast=z.sum()*0
   if kind!='none':
    u=proj(z[:,:,3,3]);contrast,valid=hbd_loss(u,st,tc,wt,class_conditional=kind=='hbd');valid_batches+=bool(valid)
   loss=cls+.1*contrast
   if not torch.isfinite(loss):nan=True;raise RuntimeError('non-finite loss')
   opt.zero_grad();loss.backward();gm+=sum(float(v.grad.norm()) for v in m.parameters() if v.grad is not None);gp+=sum(float(v.grad.norm()) for v in proj.parameters() if v.grad is not None);opt.step();tot+=float(loss)*len(x);clt+=float(cls)*len(x);ct+=float(contrast)*len(x);n+=len(x)
  logs.append({'epoch':ep+1,'loss':tot/n,'classification_loss':clt/n,'discrimination_loss':ct/n,'gamma':float(m.gamma),'mbrd_grad_sum':gm,'projection_grad_sum':gp,'valid_prototype_batches':valid_batches});print(a.variant,ep+1,tot/n,clt/n,ct/n,float(m.gamma),flush=True)
 ck={'variant':a.variant,'mode':mode,'state_dict':m.state_dict(),'projection_state_dict':proj.state_dict() if kind!='none' else None,'config':{'loss':kind,'margin':.1,'tau':.1,'lambda':.1,'projection_dim':32,'hard_B_weight':.25,'epochs':a.epochs,'batch':a.batch,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'seed':a.seed,'checkpoint':'last.pt'},'cache_sha256':sha(a.cache),'prototype_sha256':sha(a.prototypes),'e0_sha256':sha(a.weights),'initial_identity_max_abs':identity};path=a.output/'last.pt';torch.save(ck,path)
 with (a.output/'training_log.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=logs[0],lineterminator='\n');w.writeheader();w.writerows(logs)
 with torch.no_grad():
  x,mask,t,st,tc,wt=next(iter(dl));delta=float((tower_masked(tower,m(x)*mask[:,None],mask)[:,:,3,3]-tower_masked(tower,x,mask)[:,:,3,3]).abs().mean());collapse=None
  if kind!='none':
   u=proj((m(x)*mask[:,None])[:,:,3,3]);collapse={'dimension_std_mean':float(u.std(0).mean()),'mean_pair_cosine':float((u@u.T)[~torch.eye(len(u),dtype=bool)].mean())}
 rep={'variant':a.variant,'mode':mode,'loss_kind':kind,'samples':len(ds),'positive':int((typ==0).sum()),'easy':int((typ==1).sum()),'hard_B':int((typ==2).sum()),'human_confirmed_A':0,'trainable_parameters':sum(v.numel() for v in params),'inference_adapter_parameters':sum(v.numel() for v in m.parameters()),'projection_parameters_training_only':sum(v.numel() for v in proj.parameters()) if kind!='none' else 0,'initial_identity_max_abs':identity,'final_gamma':logs[-1]['gamma'],'final_loss':logs[-1]['loss'],'final_classification_loss':logs[-1]['classification_loss'],'final_discrimination_loss':logs[-1]['discrimination_loss'],'mean_abs_p3_logit_change':delta,'feature_collapse_audit':collapse,'nan':nan,'elapsed_seconds':time.time()-start,'checkpoint':str(path.resolve()),'checkpoint_sha256':sha(path),'checkpoint_bytes':path.stat().st_size};(a.output/'run.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
