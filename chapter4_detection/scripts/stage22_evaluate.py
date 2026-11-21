#!/usr/bin/env python3
"""Candidate-level CIR paired error audit (formal AP uses evaluate_original.py)."""
import argparse,csv,json
from pathlib import Path
import numpy as np
def iou(a,b):
 lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);q=np.maximum(rb-lt,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return q/(aa+bb-q+1e-12)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--refined',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--images-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();base=json.loads(a.base.read_text())['images'];ref=json.loads(a.refined.read_text())['images'];allowed={Path(x).name for x in a.images_list.read_text().splitlines()};gt=[r for r in csv.DictReader(a.gt_stats.open()) if r['image'] in allowed];rows=[];bg0=bg1=dup0=dup1=0
 for g in gt:
  name=g['image'];c=int(g['class_id']);b=np.array([float(g[x]) for x in ('x1','y1','x2','y2')]);p0=[x for x in base[name]['predictions'] if int(x['class_id'])==c];p1=[x for x in ref[name]['predictions'] if int(x['class_id'])==c]
  ovs=[iou(b,np.array(x['xyxy'])) for x in p0];k=int(np.argmax(ovs)) if ovs else -1;o0=ovs[k] if k>=0 else 0;o1=iou(b,np.array(p1[k]['xyxy'])) if k>=0 else 0;rows.append({'image':name,'gt_id':g['gt_id'],'class_id':c,'tiny':min(float(g['input_width']),float(g['input_height']))<8,'local_contrast':g['local_contrast'],'iou_before':o0,'iou_after':o1,'gain75':o0<.75<=o1,'lose75':o0>=.75>o1})
 for name in allowed:
  gs=[np.array([float(g[x]) for x in ('x1','y1','x2','y2')]) for g in gt if g['image']==name]
  for key,data in ((0,base),(1,ref)):
   ps=data[name]['predictions'];bg=sum(max((iou(np.array(x['xyxy']),g) for g in gs),default=0)<.1 for x in ps);du=sum(sum(iou(np.array(x['xyxy']),g)>=.5 for x in ps)>1 for g in gs)
   if key==0:bg0+=bg;dup0+=du
   else:bg1+=bg;dup1+=du
 summary={'gt':len(rows),'mean_iou_before':float(np.mean([r['iou_before'] for r in rows])),'mean_iou_after':float(np.mean([r['iou_after'] for r in rows])),'coverage75_before':float(np.mean([r['iou_before']>=.75 for r in rows])),'coverage75_after':float(np.mean([r['iou_after']>=.75 for r in rows])),'tiny_coverage75_before':float(np.mean([r['iou_before']>=.75 for r in rows if r['tiny']])),'tiny_coverage75_after':float(np.mean([r['iou_after']>=.75 for r in rows if r['tiny']])),'gain75':int(sum(r['gain75'] for r in rows)),'lose75':int(sum(r['lose75'] for r in rows)),'background_candidates_before_after':[int(bg0),int(bg1)],'duplicate_gt_before_after':[int(dup0),int(dup1)]};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n');
 with a.output.with_suffix('.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
