#!/usr/bin/env python3
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from chapter4_detection.models.rglcd import RGLCDDetect,replace_detect_with_rglcd
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 base=YOLO(str(a.baseline));before={k:v.detach().clone() for k,v in base.model.state_dict().items()};replace_detect_with_rglcd(base.model);after=base.model.state_dict();bad=[k for k,v in before.items() if k not in after or not torch.equal(v,after[k])]
 if bad:raise RuntimeError(f'shared state changed: {bad[:5]}')
 a.output.parent.mkdir(parents=True,exist_ok=True);base.save(a.output);re=YOLO(str(a.output));assert isinstance(re.model.model[-1],RGLCDDetect)
 bad2=[k for k,v in before.items() if k not in re.model.state_dict() or not torch.equal(v,re.model.state_dict()[k])]
 if bad2:raise RuntimeError(f'reload changed shared state: {bad2[:5]}')
 report={'baseline':str(a.baseline),'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),'output':str(a.output),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'shared_tensors':len(before),'shared_identical':True,'gamma':float(re.model.model[-1].rglcd.gamma),'phi_nonzero':bool(torch.count_nonzero(re.model.model[-1].rglcd.phi.weight)),'new_tensors':sorted(set(after)-set(before))}
 a.output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
