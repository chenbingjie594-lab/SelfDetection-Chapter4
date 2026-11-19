#!/usr/bin/env python3
"""Audit historical E2 tile geometry and label retention without modifying data."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np,pandas as pd
from chapter4_detection.scripts.slicing import make_tiles,yolo_to_xyxy

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--slices',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[];tile_rows=[]
 for ip in sorted((a.dataset/'images/train').glob('*')):
  im=cv2.imread(str(ip));H,W=im.shape[:2];items=[x.split() for x in (a.dataset/'labels/train'/f'{ip.stem}.txt').read_text().splitlines() if x.strip()];boxes=yolo_to_xyxy(np.array([[float(v) for v in x[1:]] for x in items]),W,H);tiles=make_tiles(W,H,1024,128)
  tile_rows.extend({'image':ip.name,'tile_index':ti,'instances_written':sum(1 for _ in (a.slices/'labels/train'/f'{ip.stem}__x{t.x1}_y{t.y1}.txt').read_text().splitlines() if _.strip())} for ti,t in enumerate(tiles))
  for gi,(it,b) in enumerate(zip(items,boxes)):
   area=(b[2]-b[0])*(b[3]-b[1]);full=partial=kept_partial=dropped_partial=0
   for t in tiles:
    ix=max(0,min(b[2],t.x2)-max(b[0],t.x1));iy=max(0,min(b[3],t.y2)-max(b[1],t.y1));vis=ix*iy/area
    if vis>=1-1e-9:full+=1
    elif vis>0:
     partial+=1
     if vis>=.5:kept_partial+=1
     else:dropped_partial+=1
   rows.append({'image':ip.name,'gt_index':gi,'class_id':int(it[0]),'full_tile_count':full,'partial_tile_count':partial,'kept_partial_count':kept_partial,'dropped_partial_count':dropped_partial,'at_least_one_full':full>0,'duplicated_full':full>1})
 d=pd.DataFrame(rows);t=pd.DataFrame(tile_rows);d.to_csv(a.output/'e2_slice_gt_audit.csv',index=False);t.to_csv(a.output/'e2_tile_distribution.csv',index=False)
 out={'gt_count':len(d),'tile_count':len(t),'tiles_per_image':float(len(t)/d.image.nunique()),'background_tile_ratio':float((t.instances_written==0).mean()),'gt_all_fully_covered':bool(d.at_least_one_full.all()),'gt_with_multiple_full_tiles':int(d.duplicated_full.sum()),'gt_intersecting_boundary':int((d.partial_tile_count>0).sum()),'gt_with_kept_partial_label':int((d.kept_partial_count>0).sum()),'gt_with_dropped_partial_intersection':int((d.dropped_partial_count>0).sum()),'labels_written':int(t.instances_written.sum())}
 (a.output/'e2_slice_audit_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
