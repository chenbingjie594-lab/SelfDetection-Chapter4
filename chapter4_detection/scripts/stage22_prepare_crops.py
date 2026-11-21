#!/usr/bin/env python3
"""Materialize representative aligned high/low CIR ROI pairs for audit."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from chapter4_detection.scripts.stage22_common import low_reconstruction,roi,crop
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);d=json.loads(a.candidates.read_text())['candidates'];chosen=[]
 for cls in (0,1):
  sample=next((x for x in d if x.get('positive') and x['class_id']==cls),None)
  if sample is not None:chosen.append(sample)
 rows=[]
 for c in chosen:
  im=cv2.imread(c['image_path']);lo=low_reconstruction(im);r,side=roi(c['xyxy']);h=crop(im,r);l=crop(lo,r);diff=cv2.absdiff(h,l);plate=np.hstack((h,l,diff));name=f"class{c['class_id']}_{Path(c['image']).stem}.png";cv2.imwrite(str(a.output/name),plate);rows.append({'image':c['image'],'class_id':c['class_id'],'candidate_box':c['xyxy'],'roi_xyxy':r.tolist(),'roi_side':side,'high_low_shape_equal':h.shape==l.shape,'plate':name,'mean_abs_difference':float(diff.mean())})
 (a.output/'crop_alignment.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
