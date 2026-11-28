#!/usr/bin/env python3
"""Apply a frozen Stage27 calibrator to saved E0 and FSPR candidates."""
import argparse,json,math
from pathlib import Path
def cal(s,p):
 s=min(max(s,1e-6),1-1e-6);z=p['a']*math.log(s/(1-s))+p['b'];return 1/(1+math.exp(-max(-50,min(50,z))))
def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));q=x*y;return q/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-q+1e-9)
def main():
 p=argparse.ArgumentParser();p.add_argument('--e0',type=Path,required=True);p.add_argument('--fspr',type=Path,required=True);p.add_argument('--calibrator',type=Path,required=True);p.add_argument('--target-fppi',choices=('0.1','0.25','0.5'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();e=json.loads(a.e0.read_text());f=json.loads(a.fspr.read_text());c=json.loads(a.calibrator.read_text())['sources'];thr=c['FSPR']['thresholds'][a.target_fppi]['threshold'];images={};kept=0
 for name,r in f['images'].items():
  base=[]
  for q in e['images'][name]['predictions']:
   z=dict(q);z['raw_confidence']=z['confidence'];z['confidence']=cal(z['confidence'],c['E0']['classes'][str(z['class_id'])]);z['source']='E0';base.append(z)
  add=[]
  for q in r['predictions']:
   z=dict(q);z['raw_confidence']=z['confidence'];z['confidence']=cal(z['confidence'],c['FSPR']['classes'][str(z['class_id'])]);z['source']='FSPR'
   if z['confidence']>=thr and not any(b['class_id']==z['class_id'] and iou(b['xyxy'],z['xyxy'])>=.7 for b in base):add.append(z)
  kept+=len(add);images[name]={'width':r['width'],'height':r['height'],'predictions':base+add}
 out={'schema_version':1,'model':'E0+FSPR-calibrated','images_count':len(images),'images':images,'target_fppi':a.target_fppi,'threshold':thr,'fspr_kept':kept,'outer_gt_used':False};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out)+'\n');print(json.dumps({'target_fppi':a.target_fppi,'threshold':thr,'fspr_kept':kept},indent=2))
if __name__=='__main__':main()
