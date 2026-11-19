#!/usr/bin/env python3
"""Materialize Stage12 intervention prediction files from the frozen trace outputs."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--nms-json',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 nms=json.loads(a.nms_json.read_text()); rows=list(csv.DictReader(a.trace.open())); grouped=defaultdict(list)
 for r in rows:grouped[(r['model'],r['image'])].append(r)
 for model in ('e0','cbcd'):
  for thr,records in nms[model].items():
   out={'protocol':f'NMS IoU={thr}, unchanged scores/boxes, max_det=300','images':{r['image']:r for r in records}}
   (a.output_dir/f'{model}_nms{thr}_predictions.json').write_text(json.dumps(out)+'\n')
  images={}
  # P1: no suppression, fixed top-300 score-ranked eligible class candidates.
  for (m,name),q in grouped.items():
   if m!=model:continue
   top=sorted(q,key=lambda r:float(r['score']),reverse=True)[:300]
   ref=next(r for r in nms[model]['0.7'] if r['image']==name)
   images[name]={'image':name,'width':ref['width'],'height':ref['height'],'predictions':[{'class_id':int(r['class_id']),'confidence':float(r['score']),'xyxy':[float(r[k]) for k in ('x1','y1','x2','y2')]} for r in top]}
  (a.output_dir/f'{model}_no_nms_top300_predictions.json').write_text(json.dumps({'protocol':'no NMS; top 300 eligible class candidates by original score','images':images})+'\n')
if __name__=='__main__':main()
