#!/usr/bin/env python3
"""Stage24 fixed-threshold, background, size, and failure-type diagnostics."""
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
def iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);q=np.maximum(rb-lt,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return q/(aa+bb-q+1e-12)
def main():
 p=argparse.ArgumentParser();p.add_argument('--post',type=Path,required=True);p.add_argument('--pre',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--train-list',type=Path,required=True);p.add_argument('--holdout-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();post=json.loads(a.post.read_text())['images'];pre=json.loads(a.pre.read_text())['images'];allgt=list(csv.DictReader(a.gt_stats.open()));train={Path(x).name for x in a.train_list.read_text().splitlines()};hold={Path(x).name for x in a.holdout_list.read_text().splitlines()};q25=float(np.quantile([float(r['local_contrast']) for r in allgt if r['image'] in train and int(r['class_id'])==1],.25));gt=[r for r in allgt if r['image'] in hold];rows=[];bg=0
 for g in gt:
  im=g['image'];c=int(g['class_id']);gb=[float(g[x]) for x in ('x1','y1','x2','y2')];pr=[x for x in pre[im]['predictions'] if x['class_id']==c];po=[x for x in post[im]['predictions'] if x['class_id']==c];ovs=[iou(x['xyxy'],gb) for x in pr];sp=[(x,v) for x,v in zip(pr,ovs) if v>=.1] or list(zip(pr,ovs));top=max(sp,key=lambda z:z[0]['confidence']) if sp else (None,0);oracle=max(sp,key=lambda z:z[1]) if sp else (None,0);post_iou=max([iou(x['xyxy'],gb) for x in po],default=0);det=any(x['confidence']>=.25 and iou(x['xyxy'],gb)>=.5 for x in po)
  if oracle[1]<.75:cat='A_generation_insufficient'
  elif top[1]<.75:cat='B_score_misranking'
  elif post_iou<.75:cat='C_nms_or_maxdet_suppression'
  else:cat='E_correctly_retained'
  rows.append({'fold':a.fold,'image':im,'gt_id':int(g['gt_id']),'source_group':g['source_group'],'class_id':c,'category':cat,'oracle_iou':oracle[1],'top_score_iou':top[1],'post_best_iou':post_iou,'detected_conf025_iou50':det,'tiny':min(float(g['input_width']),float(g['input_height']))<8,'low_contrast_black':c==1 and float(g['local_contrast'])<=q25})
 for im,rec in post.items():
  gs=[r for r in gt if r['image']==im]
  for x in rec['predictions']:
   if x['confidence']<.25:continue
   same=[r for r in gs if int(r['class_id'])==int(x['class_id'])];best=max([iou(x['xyxy'],[float(r[z]) for z in ('x1','y1','x2','y2')]) for r in same],default=0);bg+=best<.1
 summary={'fold':a.fold,'gt':len(rows),'low_contrast_black_q25_train':q25,'decomposition':dict(Counter(r['category'] for r in rows)),'flash_score_misranking':sum(r['class_id']==0 and r['category']=='B_score_misranking' for r in rows),'black_score_misranking':sum(r['class_id']==1 and r['category']=='B_score_misranking' for r in rows),'low_contrast_black_recall':np.mean([r['detected_conf025_iou50'] for r in rows if r['low_contrast_black']]),'low_contrast_black_n':sum(r['low_contrast_black'] for r in rows),'tiny_recall':np.mean([r['detected_conf025_iou50'] for r in rows if r['tiny']]),'tiny_n':sum(r['tiny'] for r in rows),'background_fp_conf025_iou_lt010':bg}
 conv=lambda x:x.item() if isinstance(x,np.generic) else x
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'summary':summary,'rows':rows},indent=2,default=conv)+'\n');print(json.dumps(summary,indent=2,default=conv))
if __name__=='__main__':main()
