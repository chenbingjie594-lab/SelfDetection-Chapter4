#!/usr/bin/env python3
"""Export unmatched high-response E0 detections for manual review; never labels negatives."""
import argparse, json
from pathlib import Path
import cv2

def lp(p): return Path(str(p).replace('/images/','/labels/')).with_suffix('.txt')
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); q=ix*iy
    return q/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-q+1e-9)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--images-list',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--score-min',type=float,default=.05); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); pred=json.loads(a.predictions.read_text())['images']; rows=[]
    for image in map(Path,a.images_list.read_text().splitlines()):
        im=cv2.imread(str(image)); h,w=im.shape[:2]; gt=[]
        for z in lp(image).read_text().splitlines():
            c,x,y,bw,bh=map(float,z.split()); gt.append((int(c),[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
        for n,q in enumerate(pred[image.name]['predictions']):
            if q['confidence']<a.score_min: continue
            best=max([iou(q['xyxy'],b) for _,b in gt] or [0])
            if best>=.1: continue
            x1,y1,x2,y2=map(int,q['xyxy']); pad=64; xa,ya=max(0,x1-pad),max(0,y1-pad); xb,yb=min(w,x2+pad),min(h,y2+pad); crop=im[ya:yb,xa:xb].copy(); cv2.rectangle(crop,(x1-xa,y1-ya),(x2-xa,y2-ya),(0,0,255),2); name=f'{image.stem}_{n:03d}.jpg'; cv2.imwrite(str(a.output/name),crop)
            rows.append({'image':image.name,'crop':name,'class_id':q['class_id'],'confidence':q['confidence'],'xyxy':q['xyxy'],'max_gt_iou':best,'gt_distance':'unmatched_iou_lt_0.1','review_status':'UNREVIEWED','used_as_negative':False})
    (a.output/'review_manifest.json').write_text(json.dumps({'count':len(rows),'policy':'UNREVIEWED candidates are forbidden as trusted negatives','records':rows},indent=2)+'\n'); print(json.dumps({'count':len(rows),'output':str(a.output)},indent=2))
if __name__=='__main__': main()
