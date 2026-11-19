#!/usr/bin/env python3
"""Analyze raw-GT and actual TAL-expanded anchor-center coverage at imgsz=1536."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

STRIDES=(8,16,32); IMGSZ=1536
def centers(s):return (np.arange(IMGSZ//s)+.5)*s
def count_inside(box,s):
 c=centers(s);x=c[(c>box[0])&(c<box[2])];y=c[(c>box[1])&(c<box[3])];return len(x)*len(y)
def nearest(box,s):
 cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2;c=centers(s);ix=c[np.argmin(abs(c-cx))];iy=c[np.argmin(abs(c-cy))];return float(np.hypot(ix-cx,iy-cy)),float((cx/s-.5)%1),float((cy/s-.5)%1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 src={r['image_name']:r['source_group'] for r in json.loads(a.manifest.read_text())['records'] if r['split']=='train'};rows=[]
 for r in csv.DictReader(a.gt_stats.open()):
  if r['split']!='train':continue
  scale=float(r['letterbox_scale']);pad_y=(IMGSZ-float(r.get('image_height',1024) or 1024)*scale)/2 if 'image_height' in r else (IMGSZ-1024*scale)/2
  raw=np.array([float(r['x1'])*scale,float(r['y1'])*scale+pad_y,float(r['x2'])*scale,float(r['y2'])*scale+pad_y]);cx=(raw[0]+raw[2])/2;cy=(raw[1]+raw[3])/2
  # Actual installed TAL floors each side below stride_val=stride[1]=16, preserving center.
  w=max(raw[2]-raw[0],16);h=max(raw[3]-raw[1],16);eff=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
  z={'image':r['image'],'gt_id':int(r['gt_id']),'class_id':int(r['class_id']),'source_group':src[r['image']],'original_width':float(r['width']),'original_height':float(r['height']),'input_width':float(r['input_width']),'input_height':float(r['input_height']),'local_contrast':float(r['local_contrast']),'size_group':'lt8' if min(float(r['input_width']),float(r['input_height']))<8 else ('8to16' if min(float(r['input_width']),float(r['input_height']))<16 else 'ge16')}
  for s in STRIDES:
   d,ox,oy=nearest(raw,s);z.update({f'raw_inside_p{int(np.log2(s))}':count_inside(raw,s),f'tal_inside_p{int(np.log2(s))}':count_inside(eff,s),f'nearest_p{int(np.log2(s))}_px':d,f'grid_offset_x_p{int(np.log2(s))}':ox,f'grid_offset_y_p{int(np.log2(s))}':oy})
  z['raw_inside_all']=sum(z[f'raw_inside_p{i}'] for i in (3,4,5));z['tal_inside_all']=sum(z[f'tal_inside_p{i}'] for i in (3,4,5));rows.append(z)
 with (a.output_dir/'anchor_coverage_instances.csv').open('w',newline='') as f:w=csv.DictWriter(f,rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
 def agg(q,key):
  x=np.array([r[key] for r in q]);return {'n':len(q),'mean':float(x.mean()),'median':float(np.median(x)),'p_zero':float(np.mean(x==0)),'p_le1':float(np.mean(x<=1))}
 groups={'all':rows,'flash':[r for r in rows if r['class_id']==0],'black':[r for r in rows if r['class_id']==1]}
 groups.update({k:[r for r in rows if r['size_group']==k] for k in ('lt8','8to16','ge16')});q25=np.quantile([r['local_contrast'] for r in rows],.25);groups['low_contrast_q25']=[r for r in rows if r['local_contrast']<=q25]
 summary={'protocol':{'imgsz':IMGSZ,'strides':STRIDES,'tal_stride_val':16,'strict_inside':True,'low_contrast_q25':float(q25)},'groups':{g:{key:agg(q,key) for key in ['raw_inside_p3','raw_inside_p4','raw_inside_p5','raw_inside_all','tal_inside_p3','tal_inside_p4','tal_inside_p5','tal_inside_all']} for g,q in groups.items()}}
 # Source-group bootstrap for zero/<=1 proportions on raw and effective all-scale coverage.
 by=defaultdict(list)
 for r in rows:by[r['source_group']].append(r)
 rng=np.random.default_rng(20261119);gn=list(by);boot={}
 for key in ('raw_inside_all','tal_inside_all'):
  vals=[]
  for _ in range(5000):
   q=[r for g in rng.choice(gn,len(gn),replace=True) for r in by[g]];x=np.array([r[key] for r in q]);vals.append((np.mean(x==0),np.mean(x<=1)))
  z=np.array(vals);boot[key]={'p_zero_ci95':[float(x) for x in np.quantile(z[:,0],[.025,.975])],'p_le1_ci95':[float(x) for x in np.quantile(z[:,1],[.025,.975])]}
 summary['source_group_bootstrap']=boot;(a.output_dir/'anchor_coverage_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(1,2,figsize=(9,4));ax[0].hist([r['raw_inside_all'] for r in rows],bins=np.arange(-.5,16.5),label='raw GT');ax[0].hist([r['tal_inside_all'] for r in rows],bins=np.arange(-.5,16.5),alpha=.6,label='TAL floor');ax[0].legend();ax[0].set(xlabel='inside anchors (all levels)',ylabel='GT count');ax[1].scatter([min(r['input_width'],r['input_height']) for r in rows],[r['tal_inside_all'] for r in rows],s=10,alpha=.6);ax[1].set(xlabel='minimum input side (px)',ylabel='TAL-effective inside anchors');fig.tight_layout();fig.savefig(a.output_dir/'anchor_coverage_distribution.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
