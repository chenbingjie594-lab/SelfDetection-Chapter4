#!/usr/bin/env python3
"""Paired per-GT geometry diagnosis for Matched-E0 and SFDFM predictions."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import cv2
import numpy as np

def gt_rows(label: Path, w: int, h: int):
    rows=[]
    for line in label.read_text().splitlines():
        z=line.split(); c=int(z[0]); x,y,bw,bh=map(float,z[1:])
        rows.append((c,[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
    return rows

def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/(aa+bb-inter+1e-12)

def best(gt,c,preds):
    candidates=[p for p in preds if int(p['class_id'])==c]
    if not candidates:return None,0.0
    p=max(candidates,key=lambda q:iou(gt,q['xyxy']));return p,iou(gt,p['xyxy'])

def geom(gt,p):
    if p is None:return dict(center_error=None,width_error=None,height_error=None,area_error=None,confidence=None)
    b=p['xyxy']; gw=gt[2]-gt[0];gh=gt[3]-gt[1];pw=b[2]-b[0];ph=b[3]-b[1]
    gc=((gt[0]+gt[2])/2,(gt[1]+gt[3])/2);pc=((b[0]+b[2])/2,(b[1]+b[3])/2)
    return dict(center_error=math.hypot(pc[0]-gc[0],pc[1]-gc[1])/(math.hypot(gw,gh)+1e-12),width_error=abs(pw-gw)/(gw+1e-12),height_error=abs(ph-gh)/(gh+1e-12),area_error=(pw*ph-gw*gh)/(gw*gh+1e-12),confidence=p['confidence'])

def main():
    a=argparse.ArgumentParser();a.add_argument('--baseline',type=Path,required=True);a.add_argument('--sfdfm',type=Path,required=True);a.add_argument('--labels',type=Path,required=True);a.add_argument('--images',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True);a.add_argument('--seed',type=int,required=True);x=a.parse_args()
    b=json.loads(x.baseline.read_text())['images'];s=json.loads(x.sfdfm.read_text())['images'];rows=[]
    if set(b)!=set(s):raise RuntimeError('Image sets differ')
    for name in sorted(b):
        im=cv2.imread(str(x.images/name));h,w=im.shape[:2]
        for j,(c,gt) in enumerate(gt_rows(x.labels/f'{Path(name).stem}.txt',w,h)):
            bp,bi=best(gt,c,b[name]['predictions']);sp,si=best(gt,c,s[name]['predictions'])
            row={'seed':x.seed,'image':name,'gt_index':j,'class_id':c,'gt_x1':gt[0],'gt_y1':gt[1],'gt_x2':gt[2],'gt_y2':gt[3],'gt_area':(gt[2]-gt[0])*(gt[3]-gt[1]),'border_distance':min(gt[0],gt[1],w-gt[2],h-gt[3]),'e0_iou':bi,'sfdfm_iou':si}
            for key,val in geom(gt,bp).items():row['e0_'+key]=val
            for key,val in geom(gt,sp).items():row['sfdfm_'+key]=val
            row['status50']=('both' if bi>=.5 and si>=.5 else 'e0_only' if bi>=.5 else 'sfdfm_only' if si>=.5 else 'neither')
            row['status75']=('both' if bi>=.75 and si>=.75 else 'e0_only' if bi>=.75 else 'sfdfm_only' if si>=.75 else 'neither')
            rows.append(row)
    x.output_dir.mkdir(parents=True,exist_ok=True);out=x.output_dir/f'localization_instances_seed{x.seed}.csv'
    with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    summary={}
    for t in (.5,.75,.9):summary[str(t)]={'e0':sum(r['e0_iou']>=t for r in rows),'sfdfm':sum(r['sfdfm_iou']>=t for r in rows)}
    for subset,select in {'common50':lambda r:r['e0_iou']>=.5 and r['sfdfm_iou']>=.5,'common75':lambda r:r['e0_iou']>=.75 and r['sfdfm_iou']>=.75}.items():
        z=[r for r in rows if select(r)];summary[subset]={'n':len(z)}
        for m in ('e0','sfdfm'):
            for q in ('center_error','width_error','height_error','area_error','confidence'):
                v=[r[m+'_'+q] for r in z if r[m+'_'+q] is not None];summary[subset][m+'_'+q]=float(np.mean(v)) if v else None
    (x.output_dir/f'localization_summary_seed{x.seed}.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
