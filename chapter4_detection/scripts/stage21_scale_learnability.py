#!/usr/bin/env python3
"""Train-internal grouped probe of deployable signed scale-residual features."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def iou(a,b):
 lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);q=np.maximum(rb-lt,0).prod();aa=np.maximum(a[2:]-a[:2],0).prod();bb=np.maximum(b[2:]-b[:2],0).prod();return q/(aa+bb-q+1e-12)
def corrected(box,z):
 x1,y1,x2,y2=box;cx=(x1+x2)/2;cy=(y1+y2)/2;w=(x2-x1)*math.exp(float(np.clip(z[0],-.5,.5)));h=(y2-y1)*math.exp(float(np.clip(z[1],-.5,.5)));return np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
def main():
 p=argparse.ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--train-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=5000);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);allowed={Path(x).name for x in a.train_list.read_text().splitlines() if x.strip()}
 gt=defaultdict(list)
 for r in csv.DictReader(a.gt_stats.open()):
  if r['split']=='train' and r['image'] in allowed:gt[(r['image'],int(r['class_id']))].append(r)
 cand=defaultdict(list);background=0
 for r in csv.DictReader(a.features.open()):
  if r['model']=='e0' and r['reason']=='kept' and r['image'] in allowed:cand[(r['image'],int(r['class_id']))].append(r)
 samples=[];matched_gt=set()
 for key,cc in cand.items():
  gg=gt.get(key,[]);pairs=[]
  for ci,c in enumerate(cc):
   cb=np.array([float(c[x]) for x in ('x1','y1','x2','y2')])
   for gi,g in enumerate(gg):pairs.append((iou(cb,np.array([float(g[x]) for x in ('x1','y1','x2','y2')])),ci,gi))
  usedc=set();usedg=set()
  for ov,ci,gi in sorted(pairs,reverse=True):
   if ov<.30:break
   if ci in usedc or gi in usedg:continue
   usedc.add(ci);usedg.add(gi);c=cc[ci];g=gg[gi];matched_gt.add((g['image'],int(g['gt_id'])));pb=np.array([float(c[x]) for x in ('x1','y1','x2','y2')]);gb=np.array([float(g[x]) for x in ('x1','y1','x2','y2')]);pw,ph=pb[2]-pb[0],pb[3]-pb[1];gw,gh=gb[2]-gb[0],gb[3]-gb[1];stride={'P3':8,'P4':16,'P5':32}[c['level']]
   x=np.array([math.log(max(pw*.375/stride,1e-6)),math.log(max(ph*.375/stride,1e-6)),math.log(max(pw/ph,1e-6)),math.log(max(pw*ph*.375*.375/(stride*stride),1e-6)),float(c['score']),float(c['dfl_entropy_mean']),float(c['dfl_variance_mean']),float(c['dfl_maxprob_mean']),float(c['distance_lr_asymmetry']),float(c['distance_tb_asymmetry']),int(c['class_id']),int(c['level']=='P4'),int(c['level']=='P5')])
   y=np.log(np.array([gw/pw,gh/ph]));samples.append({'c':c,'g':g,'pb':pb,'gb':gb,'x':x,'y':y,'iou0':ov})
  background+=len(cc)-len(usedc)
 total_gt=sum(len(v) for v in gt.values());X=np.stack([s['x'] for s in samples]);Y=np.stack([s['y'] for s in samples]);groups=np.array([s['g']['source_group'] for s in samples]);folds=list(GroupKFold(5).split(X,Y,groups))
 models={'linear':make_pipeline(StandardScaler(),LinearRegression()),'ridge':make_pipeline(StandardScaler(),Ridge(alpha=10.0)),'histgb':MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1.0,random_state=42))}
 pred={m:np.zeros_like(Y) for m in ('zero','median',*models)};fold_id=np.zeros(len(samples),int)
 foldrows=[]
 for fi,(tr,te) in enumerate(folds):
  fold_id[te]=fi;pred['median'][te]=np.median(Y[tr],axis=0)
  for name,model in models.items():model.fit(X[tr],Y[tr]);pred[name][te]=model.predict(X[te])
  for name in pred:
   mae=np.mean(np.abs(pred[name][te]-Y[te]),axis=0);foldrows.append({'fold':fi,'model':name,'n':len(te),'width_mae':mae[0],'height_mae':mae[1],'combined_mae':mae.mean()})
 rows=[];summary={'scope':'Fold0 training images only; frozen E0 OOF candidate features; source-group internal CV','candidate_rule':'E0 frozen raw candidate, reason=kept; greedy class-aware IoU>=0.30 matching only supplies supervision','images':len(allowed),'total_gt':total_gt,'matched_gt':len(matched_gt),'unmatched_gt':total_gt-len(matched_gt),'kept_background_candidates':background,'models':{},'folds':foldrows}
 for name,pp in pred.items():
  before=np.array([s['iou0'] for s in samples]);after=np.array([iou(s['gb'],corrected(s['pb'],z)) for s,z in zip(samples,pp)]);err=pp-Y
  q={'width_mae':float(np.mean(abs(err[:,0]))),'height_mae':float(np.mean(abs(err[:,1]))),'combined_mae':float(np.mean(abs(err))),'width_rmse':float(np.sqrt(np.mean(err[:,0]**2))),'height_rmse':float(np.sqrt(np.mean(err[:,1]**2))),'width_corr':float(np.corrcoef(pp[:,0],Y[:,0])[0,1]) if np.std(pp[:,0]) else None,'height_corr':float(np.corrcoef(pp[:,1],Y[:,1])[0,1]) if np.std(pp[:,1]) else None,'iou_before':float(before.mean()),'iou_after':float(after.mean()),'coverage75_before':float(np.mean(before>=.75)),'coverage75_after':float(np.mean(after>=.75)),'degraded_accurate':int(np.sum((before>=.75)&(after<.75))),'invalid_boxes':0}
  for sub,sel in [('Flash',np.array([int(s['g']['class_id'])==0 for s in samples])),('Black',np.array([int(s['g']['class_id'])==1 for s in samples])),('tiny',np.array([min(float(s['g']['input_width']),float(s['g']['input_height']))<8 for s in samples]))]:q[sub]={'n':int(sel.sum()),'mae':float(np.mean(abs(err[sel]))),'iou_delta':float(np.mean(after[sel]-before[sel])),'coverage75_delta':float(np.mean(after[sel]>=.75)-np.mean(before[sel]>=.75))}
  summary['models'][name]=q
  for i,s in enumerate(samples):rows.append({'image':s['g']['image'],'gt_id':s['g']['gt_id'],'source_group':s['g']['source_group'],'class_id':s['g']['class_id'],'tiny':min(float(s['g']['input_width']),float(s['g']['input_height']))<8,'fold':fold_id[i],'model':name,'target_log_w':Y[i,0],'target_log_h':Y[i,1],'pred_log_w':pp[i,0],'pred_log_h':pp[i,1],'iou_before':before[i],'iou_after':after[i]})
 # Compare the three predeclared fixed probes; no hyperparameter is retuned.
 best=min(models,key=lambda n:summary['models'][n]['combined_mae']);gn=np.unique(groups);rng=np.random.default_rng(20261121);dist=[]
 for _ in range(a.bootstrap):
  chosen=rng.choice(gn,len(gn),replace=True);idx=np.concatenate([np.where(groups==g)[0] for g in chosen]);dist.append(np.mean(abs(pred['median'][idx]-Y[idx]))-np.mean(abs(pred[best][idx]-Y[idx])))
 summary['bootstrap']={'comparison':f'{best} improvement over fold-local median MAE','repeats':a.bootstrap,'mean':float(np.mean(dist)),'ci95':np.quantile(dist,[.025,.975]).tolist(),'p_gt_0':float(np.mean(np.array(dist)>0))}
 bm=summary['models']['median'];qq=summary['models'][best];foldtrend=[r for r in foldrows if r['model']==best];medfold={r['fold']:r for r in foldrows if r['model']=='median'};good=sum(r['combined_mae']<medfold[r['fold']]['combined_mae'] for r in foldtrend)
 summary['gate']={'rule':'best fixed deployable probe >=10% lower combined MAE than fold-local median; both axes improve; >=4/5 folds; IoU and coverage improve; tiny not degraded','candidate':best,'mae_improvement':float(1-qq['combined_mae']/bm['combined_mae']),'width_improved':bool(qq['width_mae']<bm['width_mae']),'height_improved':bool(qq['height_mae']<bm['height_mae']),'positive_folds':int(good),'iou_improved':bool(qq['iou_after']>qq['iou_before']),'coverage_improved':bool(qq['coverage75_after']>qq['coverage75_before']),'tiny_not_degraded':bool(qq['tiny']['iou_delta']>=0 and qq['tiny']['coverage75_delta']>=0)}
 summary['gate']['pass']=all([summary['gate']['mae_improvement']>=.10,summary['gate']['width_improved'],summary['gate']['height_improved'],good>=4,summary['gate']['iou_improved'],summary['gate']['coverage_improved'],summary['gate']['tiny_not_degraded']])
 with (a.output/'signed_scale_residual_predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 with (a.output/'scale_group_cv_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(foldrows[0]),lineterminator='\n');w.writeheader();w.writerows(foldrows)
 with (a.output/'scale_group_bootstrap.csv').open('w',newline='') as f:w=csv.writer(f,lineterminator='\n');w.writerow(['replicate','median_minus_best_mae']);w.writerows(enumerate(dist))
 (a.output/'scale_learnability_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
