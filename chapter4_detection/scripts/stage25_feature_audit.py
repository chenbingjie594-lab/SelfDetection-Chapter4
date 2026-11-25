#!/usr/bin/env python3
"""Audit train-cache score separation and holdout output discrimination."""
import argparse,copy,json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import roc_auc_score,average_precision_score
from ultralytics import YOLO
from chapter4_detection.models.mbrd import MBRD
from chapter4_detection.models.hbd import HBDProjection
from chapter4_detection.scripts.stage25_train_hbd import tower_masked
from chapter4_detection.scripts.stage25_evaluate import iou
V={'H1':'matched','H2':'fixed','H3':'matched','H4':'matched'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('chapter4_detection/results/stage25'));p.add_argument('--stage10',type=Path,default=Path('chapter4_detection/results/stage10'));p.add_argument('--stage24',type=Path,default=Path('chapter4_detection/results/stage24'));p.add_argument('--gt-stats',type=Path,required=True);a=p.parse_args();train={v:[] for v in V};proj={v:[] for v in ('H2','H3','H4')}
 for f in range(3):
  d=torch.load(a.root/f'cache/fold{f}.pt',map_location='cpu',weights_only=False);pd=torch.load(a.stage24/f'prototypes/fold{f}.pt',map_location='cpu',weights_only=False);base=YOLO(str(a.stage10/f'runs/e0_fold{f}_seed42/weights/last.pt')).model.cpu().eval();tower=copy.deepcopy(base.model[-1].cv3[0]).eval().requires_grad_(False)
  for v,mode in V.items():
   ck=torch.load(a.root/f'runs/fold{f}_{v}/last.pt',map_location='cpu',weights_only=False);m=MBRD(d['p3_shape'][0],pd['prototypes'],mode);m.load_state_dict(ck['state_dict']);m.eval();ph=HBDProjection(d['p3_shape'][0],32) if ck['projection_state_dict'] else None
   if ph:ph.load_state_dict(ck['projection_state_dict']);ph.eval()
   scores=[];uu=[]
   with torch.no_grad():
    for s in range(0,len(d['patches']),512):
     x=d['patches'][s:s+512].float();mask=d['patch_masks'][s:s+512];z=m(x)*mask[:,None];log=tower_masked(tower,z,mask)[:,:,3,3].sigmoid();tc=d['target_class'][s:s+512];scores.append(torch.where(tc>=0,log.gather(1,tc.clamp_min(0)[:,None])[:,0],log.max(1).values));uu.append(ph(z[:,:,3,3]) if ph else torch.empty((len(x),0)))
   score=torch.cat(scores);typ=d['sample_type'];train[v].append({'fold':f,**{name:{'n':int(ms.sum()),'mean':float(score[ms].mean()),'median':float(score[ms].median())} for name,ms in [('positive',typ==0),('easy',typ==1),('hard_B',typ==2)]}})
   if ph:
    u=torch.cat(uu);marg=[]
    for c in (0,1):
     pm=(typ==0)&(d['target_class']==c);nm=(typ==2)&(d['target_class']==c)
     if pm.sum()>1 and nm.sum():
      pp=torch.nn.functional.normalize(u[pm].mean(0),dim=0);nn=torch.nn.functional.normalize((u[nm]*d['sample_weight'][nm,None]).sum(0),dim=0);marg.append({'class':c,'positive_to_positive':float((u[pm]@pp).mean()),'positive_to_hard':float((u[pm]@nn).mean()),'hard_to_hard':float((u[nm]@nn).mean()),'hard_to_positive':float((u[nm]@pp).mean())})
    proj[v].append({'fold':f,'margins':marg})
 import csv
 gt=[r for r in csv.DictReader(a.gt_stats.open()) if r['split']=='train'];hold={}
 paths={'E0':a.stage10/'e0_oof_seed42_predictions.json','M2':a.stage24/'fixed_oof_predictions.json','M3':a.stage24/'matched_oof_predictions.json',**{v:a.root/f'{v.lower()}_oof_predictions.json' for v in V}}
 for n,pth in paths.items():
  ims=json.loads(pth.read_text())['images'];pos=[];neg=[]
  for g in gt:
   gb=[float(g[x]) for x in ('x1','y1','x2','y2')];same=[p['confidence'] for p in ims[g['image']]['predictions'] if int(p['class_id'])==int(g['class_id']) and iou(p['xyxy'],gb)>=.1];pos.append(max(same,default=0))
  for im,rec in ims.items():
   gs=[g for g in gt if g['image']==im]
   for q in rec['predictions']:
    if max([iou(q['xyxy'],[float(g[x]) for x in ('x1','y1','x2','y2')]) for g in gs],default=0)<.1:neg.append(q['confidence'])
  y=np.r_[np.ones(len(pos)),np.zeros(len(neg))];s=np.r_[pos,neg];hold[n]={'positive_gt':len(pos),'background_predictions':len(neg),'positive_score_mean':float(np.mean(pos)),'background_score_mean':float(np.mean(neg)),'roc_auc':float(roc_auc_score(y,s)),'pr_auc':float(average_precision_score(y,s)),'limitation':'post-NMS candidate-dependent negative set; diagnostic only'}
 out={'training_cache':train,'training_projection_margins':proj,'holdout_output_discrimination':hold};(a.root/'feature_discrimination.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(hold,indent=2))
if __name__=='__main__':main()
