#!/usr/bin/env python3
"""Create CBCD from the exact E0 state without performing a BN-mutating forward."""
import argparse,hashlib,json,random
from pathlib import Path
import numpy as np,torch
from ultralytics import YOLO
from chapter4_detection.models.cbcd import CBCDDetect,replace_detect_with_cbcd
def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=42);a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 base=YOLO(str(a.baseline));before={k:v.detach().clone() for k,v in base.model.state_dict().items()};replace_detect_with_cbcd(base.model);after=base.model.state_dict()
 bad=[k for k,v in before.items() if k not in after or v.shape!=after[k].shape or not torch.equal(v,after[k])]
 if bad:raise RuntimeError(f'shared initialization changed: {bad[:10]}')
 a.output.parent.mkdir(parents=True,exist_ok=True);base.save(a.output);re=YOLO(str(a.output));assert isinstance(re.model.model[-1],CBCDDetect)
 loaded=re.model.state_dict();bad2=[k for k,v in before.items() if k not in loaded or v.shape!=loaded[k].shape or not torch.equal(v,loaded[k])]
 if bad2:raise RuntimeError(f'save/reload changed shared state: {bad2[:10]}')
 report={'baseline':str(a.baseline.resolve()),'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),'output':str(a.output.resolve()),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'seed':a.seed,'shared_tensors':len(before),'shared_identical_before_save':True,'shared_identical_after_reload':True,'cbcd_new_tensors':sorted(set(after)-set(before))}
 a.output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
