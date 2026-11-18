#!/usr/bin/env python3
"""Frozen-data geometry/appearance/annotation audit and grouped CV manifest builder."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,random,xml.etree.ElementTree as ET
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np

EXPECTED="48054c137c8e093957c57eef6d761750cfac6f7b437e089cebe42053fb4b62ab"
NAMES={0:"Flash point",1:"Big black spots"}
def box_iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);i=max(0,x2-x1)*max(0,y2-y1)
 aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return i/(aa+bb-i+1e-12)
def labels(rec):
 out=[]
 for i,line in enumerate(Path(rec['label']).read_text().splitlines()):
  z=line.split();c=int(z[0]);x,y,w,h=map(float,z[1:]);W,H=rec['width'],rec['height']
  out.append((i,c,((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H),line))
 return out
def roi_metrics(gray,b):
 H,W=gray.shape;x1,y1,x2,y2=map(lambda x:int(round(x)),b);x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
 t=gray[y1:y2,x1:x2];pad=max(2,int(.5*max(x2-x1,y2-y1)));a=max(0,x1-pad);q=max(0,y1-pad);c=min(W,x2+pad);d=min(H,y2+pad)
 ctx=gray[q:d,a:c];mask=np.ones(ctx.shape,bool);mask[y1-q:y2-q,x1-a:x2-a]=False;ring=ctx[mask]
 if not t.size or not ring.size:return dict(local_brightness_difference=np.nan,local_contrast=np.nan,texture_entropy=np.nan,background_gradient=np.nan,texture_similarity=np.nan)
 hist=np.histogram(t,bins=32,range=(0,256))[0].astype(float);p=hist/(hist.sum()+1e-12);ent=float(-(p[p>0]*np.log2(p[p>0])).sum())
 gx=cv2.Sobel(ctx,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(ctx,cv2.CV_32F,0,1,ksize=3)
 mt,st=float(t.mean()),float(t.std());mr,sr=float(ring.mean()),float(ring.std())
 return dict(local_brightness_difference=abs(mt-mr)/255,local_contrast=abs(mt-mr)/(sr+1e-6),texture_entropy=ent,background_gradient=float(np.hypot(gx,gy)[mask].mean()/255),texture_similarity=1/(1+abs(st-sr)/255+abs(mt-mr)/255))
def xml_boxes(path):
 root=ET.parse(path).getroot();sz=root.find('size');wh=(int(sz.findtext('width')),int(sz.findtext('height')));out=[]
 for o in root.findall('object'):
  b=o.find('bndbox');out.append((o.findtext('name'),tuple(float(b.findtext(k)) for k in ('xmin','ymin','xmax','ymax'))))
 return wh,out
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--seed',type=int,default=20261118);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 m=json.loads(a.manifest.read_text());assert m['protocol_sha256']==EXPECTED
 rows=[];issues=[];xmltrace=[]
 for rec in m['records']:
  im=cv2.imread(rec['image']);assert im is not None;H,W=im.shape[:2];gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);ls=labels(rec)
  seen=set()
  for i,c,b,raw in ls:
   x1,y1,x2,y2=b;w=x2-x1;h=y2-y1;scale=min(1536/W,1536/H);mw=round(W*scale);mh=round(H*scale);met=roi_metrics(gray,b)
   row=dict(split=rec['split'],image=rec['image_name'],gt_id=i,class_id=c,class_name=NAMES.get(c,'INVALID'),source_group=rec['source_group'],provenance=rec['provenance']['method'],x1=x1,y1=y1,x2=x2,y2=y2,width=w,height=h,area=w*h,aspect_ratio=w/(h+1e-12),relative_area=w*h/(W*H),border_distance=min(x1,y1,W-x2,H-y2),letterbox_scale=scale,input_width=w*scale,input_height=h*scale,p3_width=w*scale/8,p3_height=h*scale/8,p4_width=w*scale/16,p4_height=h*scale/16,p5_width=w*scale/32,p5_height=h*scale/32,**met)
   rows.append(row)
   if c not in (0,1):issues.append(dict(image=rec['image_name'],gt_id=i,type='invalid_class'))
   if not (0<=x1<x2<=W and 0<=y1<y2<=H):issues.append(dict(image=rec['image_name'],gt_id=i,type='invalid_or_out_of_bounds'))
   key=tuple(round(v,6) for v in (c,*b));
   if key in seen:issues.append(dict(image=rec['image_name'],gt_id=i,type='duplicate_box'))
   seen.add(key)
   if min(w,h)<2:issues.append(dict(image=rec['image_name'],gt_id=i,type='suspicious_tiny'))
   if max(w/h,h/w)>20:issues.append(dict(image=rec['image_name'],gt_id=i,type='extreme_aspect_ratio'))
  for i,(ii,ci,bi,_) in enumerate(ls):
   for jj,cj,bj,_ in ls[i+1:]:
    if ci==cj and box_iou(bi,bj)>.8:issues.append(dict(image=rec['image_name'],gt_id=f'{ii},{jj}',type='high_overlap',iou=box_iou(bi,bj)))
  ev=Path(rec['provenance'].get('evidence',''))
  if rec['provenance']['method']=='CONFIRMED_ORIGINAL_XML' and ev.exists():
   try:
    wh,xb=xml_boxes(ev); yb=[z[2] for z in ls];best=[]
    for _,x in xb:best.append(max((box_iou(x,y) for y in yb),default=0))
    xmltrace.append(dict(image=rec['image_name'],xml=str(ev),xml_width=wh[0],xml_height=wh[1],image_width=W,image_height=H,xml_boxes=len(xb),yolo_boxes=len(yb),min_best_iou=min(best) if best else None,mean_best_iou=float(np.mean(best)) if best else None,status='TRACED'))
   except Exception as e:xmltrace.append(dict(image=rec['image_name'],xml=str(ev),status='PARSE_ERROR',error=str(e)))
  else:xmltrace.append(dict(image=rec['image_name'],xml=str(ev),status='UNKNOWN'))
 def writecsv(name,data):
  if not data:return
  keys=[]
  for r in data:
   for k in r:
    if k not in keys:keys.append(k)
  with (a.output_dir/name).open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(data)
 writecsv('gt_statistics.csv',rows);writecsv('annotation_issues.csv',issues);writecsv('xml_trace.csv',xmltrace)
 # Training-only fixed difficulty thresholds.
 tr=[r for r in rows if r['split']=='train'];thr={'input_area_median':float(np.median([r['input_width']*r['input_height'] for r in tr])),'contrast_median':float(np.nanmedian([r['local_brightness_difference'] for r in tr])),'gradient_median':float(np.nanmedian([r['background_gradient'] for r in tr]))}
 # Grouped greedy three-fold assignment, frozen without model outcomes.
 groups=defaultdict(list)
 for r in tr:groups[r['source_group']].append(r)
 rng=random.Random(a.seed); names=list(groups);rng.shuffle(names);names.sort(key=lambda g:-len(groups[g]))
 folds=[[] for _ in range(3)];stats=[Counter() for _ in range(3)]
 totals=Counter();
 for r in tr:totals[f'c{r["class_id"]}']+=1;totals['small']+=(r['input_width']*r['input_height']<=thr['input_area_median']);totals['low']+=(r['local_brightness_difference']<=thr['contrast_median'])
 target={k:v/3 for k,v in totals.items()}
 random.Random(a.seed).shuffle(names)
 names.sort(key=lambda g:len(groups[g]),reverse=True)
 for g in names:
  gs=Counter();
  for r in groups[g]:gs[f'c{r["class_id"]}']+=1;gs['small']+=(r['input_width']*r['input_height']<=thr['input_area_median']);gs['low']+=(r['local_brightness_difference']<=thr['contrast_median'])
  def score(j):
   trial=[stats[z].copy() for z in range(3)];trial[j].update(gs)
   imbalance=sum(((trial[z][k]-target[k])/(target[k]+1))**2 for z in range(3) for k in target)
   sizes=[len(folds[z])+(z==j) for z in range(3)]
   return imbalance+.2*sum((s-len(names)/3)**2 for s in sizes)/(len(names)/3)**2
  j=min(range(3),key=score);folds[j].append(g);stats[j].update(gs)
 image_by_group=defaultdict(list)
 for rec in m['records']:
  if rec['split']=='train':image_by_group[rec['source_group']].append(rec['image_name'])
 fm={'schema_version':1,'seed':a.seed,'source_protocol_sha256':EXPECTED,'thresholds':thr,'folds':[]}
 for i,gs in enumerate(folds):
  ims=sorted(x for g in gs for x in image_by_group[g]);fm['folds'].append({'fold':i,'source_groups':sorted(gs),'images':ims,'statistics':dict(stats[i])})
 payload=json.dumps(fm,sort_keys=True,separators=(',',':')).encode();fm['fold_sha256']=hashlib.sha256(payload).hexdigest();(a.output_dir/'fold_manifest.json').write_text(json.dumps(fm,indent=2)+'\n')
 # Pixel perturbation sensitivity for every GT.
 sens=[]
 for r in rows:
  b=(r['x1'],r['y1'],r['x2'],r['y2'])
  for op in ('shift_x','shift_y','width','height','left','right','top','bottom'):
   for d in (-4,-2,-1,1,2,4):
    x1,y1,x2,y2=b
    if op=='shift_x':q=(x1+d,y1,x2+d,y2)
    elif op=='shift_y':q=(x1,y1+d,x2,y2+d)
    elif op=='width':q=(x1-d/2,y1,x2+d/2,y2)
    elif op=='height':q=(x1,y1-d/2,x2,y2+d/2)
    elif op=='left':q=(x1+d,y1,x2,y2)
    elif op=='right':q=(x1,y1,x2+d,y2)
    elif op=='top':q=(x1,y1+d,x2,y2)
    else:q=(x1,y1,x2,y2+d)
    sens.append({'image':r['image'],'gt_id':r['gt_id'],'class_id':r['class_id'],'width':r['width'],'height':r['height'],'operation':op,'delta_original_px':d,'delta_input_px':d*r['letterbox_scale'],'iou':box_iou(b,q)})
 writecsv('iou_sensitivity.csv',sens)
 summary={'protocol_sha256':EXPECTED,'images':Counter(r['split'] for r in m['records']),'instances':Counter(r['split'] for r in rows),'class_train':Counter(r['class_id'] for r in tr),'thresholds':thr,'input_lt8_any':sum(min(r['input_width'],r['input_height'])<8 for r in tr),'input_lt16_any':sum(min(r['input_width'],r['input_height'])<16 for r in tr),'p3_lt1_any':sum(min(r['p3_width'],r['p3_height'])<1 for r in tr),'issues':Counter(x['type'] for x in issues),'xml_trace':Counter(x['status'] for x in xmltrace),'fold_sha256':fm['fold_sha256']}
 (a.output_dir/'audit_summary.json').write_text(json.dumps(summary,indent=2,default=dict)+'\n');print(json.dumps(summary,indent=2,default=dict))
if __name__=='__main__':main()
