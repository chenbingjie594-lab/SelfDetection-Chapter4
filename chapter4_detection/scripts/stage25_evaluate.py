#!/usr/bin/env python3
"""Unified post-NMS error taxonomy and Recall-FPPI evaluation for every model."""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
THR=[.001,.005,.01,.025,.05,.1,.15,.2,.25,.4,.5,.75]
def iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);q=np.maximum(rb-lt,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return q/(aa+bb-q+1e-12)
def classify(images,gt,conf):
 out=Counter();matched=set();correct_cls=Counter();total_cls=Counter(int(r['class_id']) for r in gt)
 # Greedy one-to-one correct matches.
 cand=[]
 for gi,g in enumerate(gt):
  gb=[float(g[x]) for x in ('x1','y1','x2','y2')]
  for pi,p in enumerate(images[g['image']]['predictions']):
   if p['confidence']>=conf and int(p['class_id'])==int(g['class_id']):cand.append((iou(p['xyxy'],gb),gi,g['image'],pi))
 usedp=set()
 for ov,gi,im,pi in sorted(cand,reverse=True):
  if ov>=.5 and gi not in matched and (im,pi) not in usedp:matched.add(gi);usedp.add((im,pi));correct_cls[int(gt[gi]['class_id'])]+=1
 for im,rec in images.items():
  gs=[g for g in gt if g['image']==im]
  for pi,p in enumerate(rec['predictions']):
   if p['confidence']<conf or (im,pi) in usedp:continue
   pc=int(p['class_id']);same=[iou(p['xyxy'],[float(g[x]) for x in ('x1','y1','x2','y2')]) for g in gs if int(g['class_id'])==pc];other=[iou(p['xyxy'],[float(g[x]) for x in ('x1','y1','x2','y2')]) for g in gs if int(g['class_id'])!=pc];sa=max(same,default=0);ot=max(other,default=0);anyv=max(sa,ot)
   if sa>=.5:kind='duplicate_prediction'
   elif ot>=.5:kind='class_confusion'
   elif sa>=.1:kind='same_class_localization_error'
   elif anyv<.1:kind='background_style_prediction'
   else:kind='unconfirmed_overlap_candidate'
   out[kind]+=1
   if sa<.1:out['stage24_legacy_sameclass_iou_lt010']+=1
 return {'recall':len(matched)/len(gt),'recall_class':{str(c):correct_cls[c]/total_cls[c] for c in total_cls},'counts':dict(out),'background_fppi':out['background_style_prediction']/len(images),'matched':len(matched)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--images-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();images=json.loads(a.predictions.read_text())['images'];names={Path(x).name for x in a.images_list.read_text().splitlines()};gt=[r for r in csv.DictReader(a.gt_stats.open()) if r['image'] in names];curves=[{'confidence':t,**classify(images,gt,t)} for t in THR];out={'protocol':{'taxonomy_priority':['correct_one_to_one','duplicate_prediction','class_confusion','same_class_localization_error','background_style_prediction','unconfirmed_overlap_candidate'],'iou_match':.5,'background_any_class_iou_max':'<0.10','thresholds':THR,'limitation':'unified post-NMS attribution; no claim of exact NMS suppression reason'},'images':len(images),'gt':len(gt),'conf025':next(x for x in curves if x['confidence']==.25),'froc':curves};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['conf025'],indent=2))
if __name__=='__main__':main()
