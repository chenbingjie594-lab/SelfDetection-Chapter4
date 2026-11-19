#!/usr/bin/env python3
"""Merge Stage15 OOF predictions and compare B0/B1/B2 candidates at GT level."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from chapter4_detection.scripts.slicing import box_iou

def load_condition(root,condition):
 images={};meta=[]
 for f in range(3):
  x=json.loads((root/f'{condition.lower()}_fold{f}_predictions.json').read_text());images.update(x['images']);meta.append({k:v for k,v in x.items() if k!='images'})
 return images,meta
def cache_candidates(root,condition,image):
 for f in range(3):
  p=root.parent/'raw_candidates'/f'{condition.lower()}_fold{f}'/f'{Path(image).stem}.npz'
  if p.exists():
   z=np.load(p)
   if 'boxes_xyxy' in z:return z['boxes_xyxy'],z['class_id'],z['confidence'],z['scores'],z['level']
   xywh=z['xywh'].astype(np.float32);boxes=np.column_stack((xywh[:,0]-xywh[:,2]/2,xywh[:,1]-xywh[:,3]/2,xywh[:,0]+xywh[:,2]/2,xywh[:,1]+xywh[:,3]/2));classes=z['best_class'];scores=z['scores'].astype(np.float32);conf=scores[np.arange(len(scores)),classes];return boxes,classes,conf,scores,z['level']+3
 raise FileNotFoundError(image)
def candidate_b0(d,image):
 g=d[d.image==image];return g[['x1','y1','x2','y2']].to_numpy(),g.class_id.to_numpy(),g.score.to_numpy(),g[['flash_score','black_score']].to_numpy(),g.level.to_numpy()
def max_iou_any(box,gt):return float(box_iou(box,gt).max()) if len(gt) else 0.

def main():
 p=argparse.ArgumentParser();p.add_argument('--multiscale',type=Path,required=True);p.add_argument('--stage12-trace',type=Path,required=True);p.add_argument('--gt-statistics',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 all_images={};eff={};
 for c in ['B1','B2']:
  images,meta=load_condition(a.multiscale,c);all_images[c]=images;eff[c]=meta
  payload={'schema_version':1,'condition':c,'protocol':meta[0]['protocol'],'images_count':len(images),'mean_seconds_per_original':float(np.mean([x['mean_seconds_per_original'] for x in meta])),'mean_preprocess_seconds':float(np.mean([x['mean_preprocess_seconds'] for x in meta])),'mean_postprocess_seconds':float(np.mean([x['mean_postprocess_seconds'] for x in meta])),'peak_gpu_memory_bytes':max(x['peak_gpu_memory_bytes'] for x in meta),'images':images}
  (a.output/f'{c.lower()}_oof_predictions.json').write_text(json.dumps(payload,indent=2)+'\n')
 gt=pd.read_csv(a.gt_statistics);gt=gt[gt.split=='train'].copy();b0=pd.read_csv(a.stage12_trace);b0=b0[b0.model=='e0'];rows=[];final_stats=[]
 for image,gti in gt.groupby('image'):
  gt_boxes=gti[['x1','y1','x2','y2']].to_numpy()
  for c in ['B0','B1','B2']:
   if c=='B0':boxes,classes,conf,scores,levels=candidate_b0(b0,image)
   else:boxes,classes,conf,scores,levels=cache_candidates(a.multiscale,c,image)
   finals=(json.loads((Path('chapter4_detection/results/stage10/e0_oof_seed42_predictions.json')).read_text())['images'][image]['predictions'] if c=='B0' else all_images[c][image]['predictions'])
   fb=np.array([x['xyxy'] for x in finals],float).reshape(-1,4);fc=np.array([x['class_id'] for x in finals]);fconf=np.array([x['confidence'] for x in finals])
   background=sum(max_iou_any(x,gt_boxes)<.1 for x in fb);duplicate=max(0,len(fb)-sum(max_iou_any(x,gt_boxes)>=.5 for x in fb))
   final_stats.append({'condition':c,'image':image,'predictions':len(fb),'background_iou_lt_01':background,'nonmatched_or_duplicate':duplicate})
   for _,r in gti.iterrows():
    box=np.array([r.x1,r.y1,r.x2,r.y2]);mask=classes==r.class_id;cb=boxes[mask];cc=conf[mask];cl=levels[mask]
    ious=box_iou(box,cb) if len(cb) else np.array([]);j=int(ious.argmax()) if len(ious) else -1;q=float(ious[j]) if j>=0 else 0.;best=cb[j] if j>=0 else np.full(4,np.nan)
    cx=lambda z:(z[0]+z[2])/2;cy=lambda z:(z[1]+z[3])/2
    rows.append({'condition':c,'image':image,'gt_id':int(r.gt_id),'source_group':r.source_group,'class_id':int(r.class_id),'tiny':bool(min(r.input_width,r.input_height)<8),'qmax':q,'covered50':q>=.5,'covered75':q>=.75,'covered90':q>=.9,'candidate_score':float(cc[j]) if j>=0 else 0.,'candidate_level':str(cl[j]) if j>=0 else '', 'candidate_x1':float(best[0]),'candidate_y1':float(best[1]),'candidate_x2':float(best[2]),'candidate_y2':float(best[3]),'gt_x1':float(box[0]),'gt_y1':float(box[1]),'gt_x2':float(box[2]),'gt_y2':float(box[3]),'center_error':float(np.hypot(cx(best)-cx(box),cy(best)-cy(box))) if j>=0 else np.nan,'width_relative_error':float(abs((best[2]-best[0])-(box[2]-box[0]))/(box[2]-box[0])) if j>=0 else np.nan,'height_relative_error':float(abs((best[3]-best[1])-(box[3]-box[1]))/(box[3]-box[1])) if j>=0 else np.nan})
 out=pd.DataFrame(rows);out.to_csv(a.output/'candidate_quality_per_gt.csv',index=False);fs=pd.DataFrame(final_stats);fs.to_csv(a.output/'final_prediction_diagnostics.csv',index=False)
 summary={}
 for c,g in out.groupby('condition'):
  summary[c]={'gt':len(g),'qmax_mean':float(g.qmax.mean()),'coverage50':float(g.covered50.mean()),'coverage75':float(g.covered75.mean()),'coverage90':float(g.covered90.mean()),'tiny_coverage75':float(g.loc[g.tiny,'covered75'].mean()),'flash_coverage75':float(g.loc[g.class_id==0,'covered75'].mean()),'black_coverage75':float(g.loc[g.class_id==1,'covered75'].mean()),'center_error_mean':float(g.center_error.mean()),'width_relative_error_mean':float(g.width_relative_error.mean()),'height_relative_error_mean':float(g.height_relative_error.mean())}
 summary['final_predictions']={c:{'per_image':float(g.predictions.mean()),'background_per_image':float(g.background_iou_lt_01.mean()),'nonmatched_or_duplicate_per_image':float(g.nonmatched_or_duplicate.mean())} for c,g in fs.groupby('condition')}
 summary['efficiency']={c:{'seconds':float(np.mean([x['mean_seconds_per_original'] for x in eff[c]])),'preprocess_seconds':float(np.mean([x['mean_preprocess_seconds'] for x in eff[c]])),'postprocess_seconds':float(np.mean([x['mean_postprocess_seconds'] for x in eff[c]])),'peak_gpu_memory_bytes':max(x['peak_gpu_memory_bytes'] for x in eff[c])} for c in eff}
 (a.output/'candidate_analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
