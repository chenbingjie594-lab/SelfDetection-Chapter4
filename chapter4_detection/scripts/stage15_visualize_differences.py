#!/usr/bin/env python3
"""Create B0/B1/B2 side-by-side candidate views for largest B1-B2 changes."""
import argparse
from pathlib import Path
import cv2,numpy as np,pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=pd.read_csv(a.csv);w=d.pivot(index=['image','gt_id'],columns='condition',values='qmax').reset_index();w['change']=(w.B1-w.B2).abs();sel=w.nlargest(12,'change');canvas=np.full((len(sel)*180,780,3),255,np.uint8)
 for ri,(_,s) in enumerate(sel.iterrows()):
  base=cv2.imread(str(a.images/s.image));records=d[(d.image==s.image)&(d.gt_id==s.gt_id)].set_index('condition')
  for ci,c in enumerate(['B0','B1','B2']):
   r=records.loc[c];x1=max(0,int(r.gt_x1)-80);y1=max(0,int(r.gt_y1)-80);x2=min(base.shape[1],int(r.gt_x2)+80);y2=min(base.shape[0],int(r.gt_y2)+80);crop=base[y1:y2,x1:x2].copy();cv2.rectangle(crop,(int(r.gt_x1)-x1,int(r.gt_y1)-y1),(int(r.gt_x2)-x1,int(r.gt_y2)-y1),(0,255,0),2)
   if np.isfinite(r.candidate_x1):cv2.rectangle(crop,(int(r.candidate_x1)-x1,int(r.candidate_y1)-y1),(int(r.candidate_x2)-x1,int(r.candidate_y2)-y1),(0,0,255),2)
   crop=cv2.resize(crop,(250,150));cv2.putText(crop,f'{c} Q={r.qmax:.3f}',(5,145),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);canvas[ri*180:ri*180+150,ci*260:ci*260+250]=crop
  cv2.putText(canvas,f'{s.image} gt{int(s.gt_id)} |B1-B2|={s.change:.3f}',(0,ri*180+172),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1)
 a.output.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(a.output),canvas)
if __name__=='__main__':main()
