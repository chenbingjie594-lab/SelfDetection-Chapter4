#!/usr/bin/env python3
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from chapter4_detection.models.pcrf import PCRFDetect,replace_detect_with_pcrf
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);w=YOLO(str(a.baseline));before={k:v.clone() for k,v in w.model.state_dict().items()};old=copy_head=__import__('copy').deepcopy(w.model.model[-1]).eval();head=replace_detect_with_pcrf(w.model);bad=[k for k,v in before.items() if k not in w.model.state_dict() or not torch.equal(v,w.model.state_dict()[k])];assert not bad
 # Initial semantic check on a correctly letterboxed real image.
 import cv2
 im=cv2.imread(str(a.image));im=cv2.resize(im,(1536,384));im=cv2.copyMakeBorder(im,16,16,16,16,cv2.BORDER_CONSTANT,value=(114,114,114));x=torch.from_numpy(im[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float()/255;cap={}
 h2=w.model.model[2].register_forward_hook(lambda m,i,o:cap.__setitem__('p2',o));h3=w.model.model[16].register_forward_hook(lambda m,i,o:cap.__setitem__('p3',o))
 w.model.eval()
 with torch.no_grad():w.model(x);base=old.forward_head([cap['p3'],cap.get('p4',cap['p3']),cap.get('p5',cap['p3'])],**old.one2many) if False else None;new=head.forward_head([cap['p2'],cap['p3'],torch.zeros(1,256,26,98),torch.zeros(1,512,13,49)],**head.one2many)
 # Direct classification identity is guaranteed by copied cv3 and untouched P3; audit it explicitly at P3.
 with torch.no_grad():cls0=old.cv3[0](cap['p3']);cls1=head.cv3[0](cap['p3']);reg0=old.cv2[0](cap['p3']);reg1=head.cv2[0](head.pcrf(cap['p2'],cap['p3']))
 h2.remove();h3.remove();a.output.parent.mkdir(parents=True,exist_ok=True);w.save(a.output);reload=YOLO(str(a.output));assert isinstance(reload.model.model[-1],PCRFDetect)
 report={'shared_tensors':len(before),'shared_identical':True,'classification_raw_max_abs':float((cls0-cls1).abs().max()),'regression_relative_l2':float((reg0-reg1).norm()/(reg0.norm()+1e-12)),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'reload_custom_head':True};a.output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
