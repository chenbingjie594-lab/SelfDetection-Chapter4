#!/usr/bin/env python3
"""Frozen E0 candidate geometry and counterfactual box repair analysis."""
import argparse,csv,json,math
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/(aa+bb-inter+1e-12)
def geom(gt,p):
 gx=(gt[0]+gt[2])/2;gy=(gt[1]+gt[3])/2;gw=gt[2]-gt[0];gh=gt[3]-gt[1];px=(p[0]+p[2])/2;py=(p[1]+p[3])/2;pw=p[2]-p[0];ph=p[3]-p[1]
 return {'center_x_rel':abs(px-gx)/gw,'center_y_rel':abs(py-gy)/gh,'center_x_orig_px':abs(px-gx),'center_y_orig_px':abs(py-gy),'center_x_input_px':abs(px-gx)*.375,'center_y_input_px':abs(py-gy)*.375,'width_abs_rel':abs(pw-gw)/gw,'height_abs_rel':abs(ph-gh)/gh,'width_signed_rel':(pw-gw)/gw,'height_signed_rel':(ph-gh)/gh,'left_orig_px':p[0]-gt[0],'top_orig_px':p[1]-gt[1],'right_orig_px':p[2]-gt[2],'bottom_orig_px':p[3]-gt[3]}
def repair(gt,p):
 gx=(gt[0]+gt[2])/2;gy=(gt[1]+gt[3])/2;gw=gt[2]-gt[0];gh=gt[3]-gt[1];px=(p[0]+p[2])/2;py=(p[1]+p[3])/2;pw=p[2]-p[0];ph=p[3]-p[1]
 boxes={'center':(gx-pw/2,gy-ph/2,gx+pw/2,gy+ph/2),'size':(px-gw/2,py-gh/2,px+gw/2,py+gh/2)}
 names='ltrb'
 for j,n in enumerate(names):q=list(p);q[j]=gt[j];boxes[n]=q
 for pair in ('lr','tb','lt','lb','rt','rb'):
  q=list(p)
  for n in pair:q[names.index(n)]=gt[names.index(n)]
  boxes[pair]=q
 return {k:iou(gt,v) if v[2]>v[0] and v[3]>v[1] else float('nan') for k,v in boxes.items()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidates',type=Path,required=True);ap.add_argument('--gt-stats',type=Path,required=True);ap.add_argument('--failures',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 cands=defaultdict(list)
 for r in csv.DictReader(a.candidates.open()):
  if r['model']=='e0':cands[(r['image'],int(r['class_id']))].append(r)
 failures={(r['image'],int(r['gt_id'])):r for r in csv.DictReader(a.failures.open()) if r['model']=='e0'}
 out=[];cf=[]
 for g in csv.DictReader(a.gt_stats.open()):
  if g['split']!='train':continue
  key=(g['image'],int(g['class_id']));gt=tuple(float(g[x]) for x in ('x1','y1','x2','y2'));items=cands[key]
  if not items:continue
  for q in items:q['_iou']=iou(gt,tuple(float(q[x]) for x in ('x1','y1','x2','y2')))
  chosen={'oracle':max(items,key=lambda x:x['_iou']),'top_score':max(items,key=lambda x:float(x['score']))}
  kept=[x for x in items if x['reason']=='kept'];chosen['nms_kept']=max(kept,key=lambda x:x['_iou']) if kept else None
  fail=failures.get((g['image'],int(g['gt_id'])),{});tiny=min(float(g['input_width']),float(g['input_height']))<8
  for typ,q in chosen.items():
   if q is None:continue
   p=tuple(float(q[x]) for x in ('x1','y1','x2','y2'));z={'image':g['image'],'gt_id':g['gt_id'],'class_id':g['class_id'],'class_name':g['class_name'],'source_group':g['source_group'],'tiny':tiny,'failure_category':fail.get('category',''),'selection':typ,'candidate_id':q['candidate_id'],'anchor_id':q['anchor_id'],'score':q['score'],'reason':q['reason'],'iou':q['_iou'],**geom(gt,p)}
   for edge in ('left','top','right','bottom'):z[edge+'_input_px']=z[edge+'_orig_px']*.375;z[edge+'_relative']=z[edge+'_orig_px']/(float(g['width']) if edge in ('left','right') else float(g['height']))
   out.append(z)
   if typ=='oracle':cf.append({'image':g['image'],'gt_id':g['gt_id'],'class_id':g['class_id'],'class_name':g['class_name'],'source_group':g['source_group'],'tiny':tiny,'failure_category':fail.get('category',''),'candidate_id':q['candidate_id'],'score':q['score'],'iou_original':q['_iou'],**repair(gt,p)})
 def write(name,rows):
  with (a.output/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 write('geometry_by_gt_selection.csv',out);write('counterfactual_box_repairs.csv',cf)
 summary={}
 for sel in ('oracle','top_score','nms_kept'):
  rr=[x for x in out if x['selection']==sel];summary[sel]=summ(rr)
 oracle=[x for x in out if x['selection']=='oracle']
 summary['oracle_subgroups']={name:summ(z) for name,z in [('Flash',[x for x in oracle if x['class_id']=='0']),('Black',[x for x in oracle if x['class_id']=='1']),('tiny',[x for x in oracle if x['tiny'] is True])]}
 summary['oracle_error_iou_correlation']={k:float(np.corrcoef([float(x[k]) for x in oracle],[float(x['iou']) for x in oracle])[0,1]) for k in ('center_x_rel','center_y_rel','width_abs_rel','height_abs_rel')}
 A=[x for x in cf if x['failure_category'].startswith('A_')];allA=[r for r in failures.values() if r['category'].startswith('A_')];summary['A_failure_count_total']=len(allA);summary['A_failure_count_with_cached_candidate']=len(A);summary['A_missing_cached_candidate']=len(allA)-len(A);summary['A_repairs']={k:{'reach75':sum(float(x[k])>=.75 for x in A),'rate_of_analyzable':sum(float(x[k])>=.75 for x in A)/len(A)} for k in ('center','size','l','t','r','b','lr','tb','lt','lb','rt','rb')};summary['A_unrepaired_analyzable']=sum(max(float(x[k]) for k in ('center','size','l','t','r','b','lr','tb','lt','lb','rt','rb'))<.75 for x in A);summary['A_tiny_analyzable']=sum(str(x['tiny'])=='True' or x['tiny'] is True for x in A);summary['A_tiny_total_from_stage12']=34;summary['A_class_analyzable']=dict(Counter(x['class_name'] for x in A));summary['A_subgroups']={name:{'n':len(z),'center_reach75':sum(float(x['center'])>=.75 for x in z),'size_reach75':sum(float(x['size'])>=.75 for x in z),'lr_reach75':sum(float(x['lr'])>=.75 for x in z),'tb_reach75':sum(float(x['tb'])>=.75 for x in z)} for name,z in [('Flash',[x for x in A if x['class_id']=='0']),('Black',[x for x in A if x['class_id']=='1']),('tiny',[x for x in A if str(x['tiny'])=='True' or x['tiny'] is True])]}
 (a.output/'geometry_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 try:
  import matplotlib.pyplot as plt
  rr=[x for x in out if x['selection']=='oracle'];fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].hist([x['center_x_rel'] for x in rr],30,alpha=.6,label='center x');ax[0].hist([x['center_y_rel'] for x in rr],30,alpha=.6,label='center y');ax[0].legend();ax[1].hist([x['width_signed_rel'] for x in rr],30,alpha=.6,label='width');ax[1].hist([x['height_signed_rel'] for x in rr],30,alpha=.6,label='height');ax[1].legend();fig.tight_layout();fig.savefig(a.output/'geometry_error_distributions.png',dpi=180);plt.close(fig)
 except Exception as e:print('plot skipped',e)
def summ(rr):
 keys=('iou','center_x_rel','center_y_rel','center_x_orig_px','center_y_orig_px','center_x_input_px','center_y_input_px','width_abs_rel','height_abs_rel','width_signed_rel','height_signed_rel','left_orig_px','top_orig_px','right_orig_px','bottom_orig_px');d={'n':len(rr)}
 for k in keys:
  v=np.array([float(x[k]) for x in rr]);d[k]={'mean':float(v.mean()),'median':float(np.median(v)),'p90_abs':float(np.quantile(abs(v),.9))}
 return d
if __name__=='__main__':main()
