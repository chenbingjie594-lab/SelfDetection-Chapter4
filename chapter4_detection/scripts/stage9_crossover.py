#!/usr/bin/env python3
"""Create C0/C1/C2/C3 predictions from strictly aligned pre-NMS candidates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression

def main():
    p=argparse.ArgumentParser();p.add_argument('--e0',type=Path,required=True);p.add_argument('--sfdfm',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    m0=json.loads((a.e0/'manifest.json').read_text());m1=json.loads((a.sfdfm/'manifest.json').read_text())
    if m0['head']!=m1['head'] or set(m0['records'])!=set(m1['records']):raise RuntimeError('Head metadata or image sets differ; crossover forbidden')
    out={k:{} for k in ('C0_B0S0','C1_B1S1','C2_B0S1','C3_B1S0')}
    for name in sorted(m0['records']):
        r0=m0['records'][name];r1=m1['records'][name]
        for key in ('input_shape','original_shape','ratio_pad','feature_shapes'):
            if r0[key]!=r1[key]:raise RuntimeError(f'{name}: {key} differs; crossover forbidden')
        z0=np.load(a.e0/r0['file']);z1=np.load(a.sfdfm/r1['file'])
        # decoded rows are xywh plus probabilities; raw arrays prove candidate ordering/count compatibility.
        if z0['decoded'].shape!=z1['decoded'].shape or z0['raw_boxes'].shape!=z1['raw_boxes'].shape or z0['logits'].shape!=z1['logits'].shape:raise RuntimeError(f'{name}: raw candidate shapes differ')
        boxes=[z0['decoded'][:4],z1['decoded'][:4]];scores=[torch.from_numpy(z0['logits']).sigmoid().numpy(),torch.from_numpy(z1['logits']).sigmoid().numpy()]
        for key,bi,si in [('C0_B0S0',0,0),('C1_B1S1',1,1),('C2_B0S1',0,1),('C3_B1S0',1,0)]:
            pred=torch.from_numpy(np.concatenate([boxes[bi],scores[si]],axis=0))[None]
            det=non_max_suppression(pred,.001,.7,nc=2,multi_label=True,agnostic=False,max_det=300)[0]
            (gy,gx),(px,py)=r0['ratio_pad'];det[:,[0,2]]=(det[:,[0,2]]-px)/gx;det[:,[1,3]]=(det[:,[1,3]]-py)/gy
            out[key][name]={"width":r0['original_shape'][1],"height":r0['original_shape'][0],"predictions":[{"class_id":int(q[5]),"confidence":float(q[4]),"xyxy":[float(v) for v in q[:4]]} for q in det]}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for key,images in out.items():(a.output_dir/f'{key}_predictions.json').write_text(json.dumps({"schema_version":1,"model":key,"protocol":"conf=.001,iou=.7,max_det=300,multi_label,class-aware","images":images},indent=2)+'\n')
if __name__=='__main__':main()
