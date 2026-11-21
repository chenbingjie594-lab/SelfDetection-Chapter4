#!/usr/bin/env python3
"""Create leakage-controlled outer-training gate targets via inner 3-fold CIR cross-fitting."""
from __future__ import annotations
import argparse,json,random,subprocess,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from chapter4_detection.models.cir import CIRRefiner,apply_residual
from chapter4_detection.models.qpr_cir import gate_features
from chapter4_detection.scripts.stage22_common import CIRCandidates,box_iou_diag

def subset(src,items,path,tag):
 d={k:v for k,v in src.items() if k!='candidates'};d['mode']=tag;d['candidates']=items;path.write_text(json.dumps(d)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--outer-fold',type=int,required=True);p.add_argument('--candidates',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--outer-holdout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--python',default=sys.executable);p.add_argument('--epochs',type=int,default=40);p.add_argument('--seed',type=int,default=42);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 import csv
 rows=list(csv.DictReader(a.gt_stats.open()));image_group={r['image']:r['source_group'] for r in rows if r['split']=='train'};src=json.loads(a.candidates.read_text());items=src['candidates'];missing=sorted({x['image'] for x in items}-image_group.keys());assert not missing,missing
 outer_holdout={Path(x).name for x in a.outer_holdout.read_text().splitlines()};leaked=sorted({x['image'] for x in items}&outer_holdout);assert not leaked,f'outer holdout leaked into gate targets: {leaked}'
 groups=sorted({image_group[x['image']] for x in items});random.Random(a.seed+a.outer_fold).shuffle(groups);assignment={g:i%3 for i,g in enumerate(groups)};records=[];fold_report=[]
 for inner in range(3):
  idir=a.output/f'inner{inner}';idir.mkdir(exist_ok=True);train=[x for x in items if assignment[image_group[x['image']]]!=inner];hold=[x for x in items if assignment[image_group[x['image']]]==inner];tp=idir/'train_candidates.json';hp=idir/'hold_candidates.json';subset(src,train,tp,'inner_train');subset(src,hold,hp,'inner_hold')
  run=idir/'cir';subprocess.run([a.python,'chapter4_detection/scripts/stage22_train_cir.py','--candidates',str(tp),'--variant','C','--output',str(run),'--epochs',str(a.epochs),'--seed',str(a.seed)],check=True)
  ck=torch.load(run/'last.pt',map_location='cpu',weights_only=False);model=CIRRefiner('C');model.load_state_dict(ck['state_dict']);model.eval();ds=CIRCandidates(hp);dl=DataLoader(ds,batch_size=32,shuffle=False,num_workers=0);offset=0;useful=harmful=neutral=0
  with torch.no_grad():
   for q in dl:
    fh=model.encoder(q['high']);fl=model.encoder(q['low']);d=model(q['high'],q['low'],q['semantic']);pr=apply_residual(q['box'],d);gt=apply_residual(q['box'],q['target']);b0=box_iou_diag(q['box'],gt);b1=box_iou_diag(pr,gt);geo=gate_features(q['semantic'],d,mode='geo');full=gate_features(q['semantic'],d,fh,fl,'full')
    for j in range(len(d)):
     c=hold[offset+j];pos=bool(q['positive'][j]);delta=float(b1[j]-b0[j]) if pos else 0.;label=float(delta>.01) if pos else 0.;useful+=int(pos and delta>.01);harmful+=int(pos and delta<-.02);neutral+=int(pos and not(delta>.01 or delta<-.02));records.append({'outer_fold':a.outer_fold,'inner_fold':inner,'candidate_id':c['candidate_id'],'image':c['image'],'source_group':image_group[c['image']],'class_id':c['class_id'],'positive':pos,'tiny':bool(c.get('positive') and min(abs(c['gt_xyxy'][2]-c['gt_xyxy'][0])*.375,abs(c['gt_xyxy'][3]-c['gt_xyxy'][1])*.375)<8),'box':q['box'][j].tolist(),'gt_box':gt[j].tolist(),'cir_residual':d[j].tolist(),'iou_before':float(b0[j]) if pos else None,'iou_cir':float(b1[j]) if pos else None,'delta_iou':delta if pos else None,'quality_label':label,'geo':geo[j].tolist(),'full':full[j].tolist()})
    offset+=len(d)
  fold_report.append({'inner_fold':inner,'train_candidates':len(train),'hold_candidates':len(hold),'train_groups':len(groups)-sum(v==inner for v in assignment.values()),'hold_groups':sum(v==inner for v in assignment.values()),'useful':useful,'harmful':harmful,'neutral':neutral})
 assert len(records)==len(items) and len({r['candidate_id'] for r in records})==len(items)
 positives=[r for r in records if r['positive']];report={'outer_fold':a.outer_fold,'protocol':'3-fold source-group inner CIR cross-fit; outer holdout absent','outer_holdout_images':len(outer_holdout),'outer_holdout_overlap':len(leaked),'candidates':len(records),'positive':len(positives),'background':len(records)-len(positives),'useful':sum(r['delta_iou']>.01 for r in positives),'harmful':sum(r['delta_iou']<-.02 for r in positives),'neutral':sum(-.02<=r['delta_iou']<=.01 for r in positives),'Flash':sum(r['class_id']==0 for r in positives),'Black':sum(r['class_id']==1 for r in positives),'tiny':sum(r['tiny'] for r in positives),'inner_folds':fold_report}
 (a.output/'quality_targets.json').write_text(json.dumps({'report':report,'records':records})+'\n');(a.output/'crossfit_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
