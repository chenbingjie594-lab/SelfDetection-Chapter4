#!/usr/bin/env python3
"""Build matched controls and audit initialization, forward, loss and gradients."""
from __future__ import annotations
import argparse, copy, hashlib, json, random
from pathlib import Path
import cv2, numpy as np, torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from chapter4_detection.models.nbr_de import NBRDE, register_nbr_de
from chapter4_detection.scripts.model_audit import semantic_transfer_e3

def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def save_model(model,path):
 y=YOLO("yolo11s.yaml");y.model=model;path.parent.mkdir(parents=True,exist_ok=True);y.save(path)
def real_batch(image,label,imgsz,device):
 im=cv2.imread(str(image));h,w=im.shape[:2];gain=min(imgsz/w,imgsz/h);nw,nh=round(w*gain),round(h*gain);r=cv2.resize(im,(nw,nh));canvas=np.full((imgsz,imgsz,3),114,np.uint8);px=(imgsz-nw)//2;py=(imgsz-nh)//2;canvas[py:py+nh,px:px+nw]=r
 rows=[z.split() for z in label.read_text().splitlines() if z.strip()]; cls=[];boxes=[]
 for z in rows:
  c,x,y,bw,bh=int(z[0]),*map(float,z[1:]);cls.append(c);boxes.append([(x*w*gain+px)/imgsz,(y*h*gain+py)/imgsz,bw*w*gain/imgsz,bh*h*gain/imgsz])
 t=torch.from_numpy(canvas[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float().div(255).to(device)
 return {"img":t,"cls":torch.tensor(cls,dtype=torch.float32,device=device).view(-1,1),"bboxes":torch.tensor(boxes,dtype=torch.float32,device=device),"batch_idx":torch.zeros(len(cls),device=device)}
def flat_outputs(x):
 if torch.is_tensor(x):return [x]
 out=[]
 if isinstance(x,dict):
  for z in x.values(): out.extend(flat_outputs(z))
  return out
 for z in x:
  if torch.is_tensor(z):out.append(z)
  elif isinstance(z,(list,tuple)):out.extend(flat_outputs(z))
 return out
def compare(a,b):
 pairs=list(zip(flat_outputs(a),flat_outputs(b)));return {"tensors":len(pairs),"max_abs":max((float((x-y).abs().max()) for x,y in pairs),default=0),"all_equal":all(torch.equal(x,y) for x,y in pairs)}
def grad_stats(module):
 out={}
 for n,p in module.named_parameters():
  g=p.grad;out[n]={"parameter_norm":float(p.detach().float().norm()),"grad_none":g is None,"grad_norm":None if g is None else float(g.detach().float().norm()),"grad_zero":None if g is None else bool(torch.count_nonzero(g)==0),"finite":None if g is None else bool(torch.isfinite(g).all())}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--pretrained",type=Path,required=True);p.add_argument("--image",type=Path,required=True);p.add_argument("--label",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--imgsz",type=int,default=640);p.add_argument("--device",default="cpu");a=p.parse_args();register_nbr_de();seed_all(a.seed);a.output_dir.mkdir(parents=True,exist_ok=True)
 src=YOLO(str(a.pretrained)).model; cfg=copy.deepcopy(src.yaml);cfg["nc"]=2;base=DetectionModel(cfg=cfg,ch=3,nc=2,verbose=False);base.load(src);m0=a.output_dir/f"matched_e0_init_seed{a.seed}.pt";save_model(base,m0)
 paths={"M0":m0}
 for name,yaml in (("MID","chapter4_detection/models/yolo11s_nbr_de_identity.yaml"),("MFULL","chapter4_detection/models/yolo11s_nbr_de.yaml")):
  seed_all(a.seed);y=YOLO(yaml);tr=semantic_transfer_e3(y,m0);path=a.output_dir/f"{name.lower()}_init_seed{a.seed}.pt";y.save(path);paths[name]=path
  if not tr["value_verification_passed"]:raise RuntimeError(f"{name} transfer failed")
 models={k:YOLO(str(v)).model.to(a.device) for k,v in paths.items()};batch=real_batch(a.image,a.label,a.imgsz,a.device)
 for m in models.values(): m.args=get_cfg()
 # Train-mode raw detect tensors and original YOLO loss.
 outputs={};losses={};grads={}
 for k,m in models.items():
  m.train();m.zero_grad(set_to_none=True); outputs[k]=m(batch["img"]); loss,items=m(batch);loss.sum().backward();losses[k]={"total":float(loss.detach().sum()),"items":{n:float(v.detach().sum()) for n,v in items.items()}};grads[k]={"module":grad_stats(next((z for z in m.modules() if isinstance(z,NBRDE)),m)),"shared":grad_stats(m)}
 shared={}
 s0=models["M0"].state_dict()
 for name,shift in (("MID",1),("MFULL",1)):
  st=models[name].state_dict();bad=[]
  for key,v in s0.items():
   head,layer,suffix=key.split(".",2);target=f"model.{int(layer) if int(layer)<=16 else int(layer)+shift}.{suffix}"
   if target not in st or not torch.equal(v,st[target]):bad.append({"source":key,"target":target})
  shared[name]={"compared":len(s0),"mismatches":bad}
 grad_equal={}
 p0=dict(models["M0"].named_parameters())
 for name in ("MID","MFULL"):
  pn=dict(models[name].named_parameters());bad=[];compared=0
  for key,pv in p0.items():
   head,layer,suffix=key.split(".",2);target=f"model.{int(layer) if int(layer)<=16 else int(layer)+1}.{suffix}"
   if target not in pn:continue
   compared+=1;g0,g1=pv.grad,pn[target].grad
   if (g0 is None)!=(g1 is None) or (g0 is not None and not torch.equal(g0,g1)):bad.append({"source":key,"target":target,"max_abs":None if g0 is None or g1 is None else float((g0-g1).abs().max())})
  grad_equal[name]={"compared":compared,"mismatches":bad}
 report={"seed":a.seed,"imgsz":a.imgsz,"device":a.device,"paths":{k:str(v.resolve()) for k,v in paths.items()},"sha256":{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},"shared_parameter_equality":shared,"shared_gradient_equality":grad_equal,"raw_output_comparison":{"M0_vs_MID":compare(outputs["M0"],outputs["MID"]),"M0_vs_MFULL":compare(outputs["M0"],outputs["MFULL"])},"losses":losses,"loss_abs_delta":{"M0_vs_MID":abs(losses["M0"]["total"]-losses["MID"]["total"]),"M0_vs_MFULL":abs(losses["M0"]["total"]-losses["MFULL"]["total"])},"gradient_stats":grads}
 (a.output_dir/f"initialization_audit_seed{a.seed}.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"shared":{k:len(v['mismatches']) for k,v in shared.items()},"outputs":report["raw_output_comparison"],"loss_delta":report["loss_abs_delta"]},indent=2))
if __name__=="__main__":main()
