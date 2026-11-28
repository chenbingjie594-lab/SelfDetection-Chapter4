#!/usr/bin/env python3
"""Evaluate Stage27 predictions when CV images span the frozen train/val label dirs."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils.metrics import DetMetrics, box_iou
from chapter4_detection.scripts.evaluate_original import load_gt, match_predictions

def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--labels-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); data=json.loads(a.predictions.read_text()); metrics=DetMetrics(names={0:'Flash point',1:'Big black spots'}); sizes=[]
    for image_index,(name,record) in enumerate(sorted(data['images'].items())):
        candidates=[a.labels_root/'train'/f'{Path(name).stem}.txt',a.labels_root/'val'/f'{Path(name).stem}.txt']; labels=next((x for x in candidates if x.exists()),None)
        if labels is None: raise FileNotFoundError(name)
        tc,tb=load_gt(labels,record['width'],record['height']); preds=record['predictions']; pc=torch.tensor([q['class_id'] for q in preds],dtype=torch.float32); pb=torch.tensor([q['xyxy'] for q in preds],dtype=torch.float32).reshape(-1,4); conf=torch.tensor([q['confidence'] for q in preds],dtype=torch.float32); correct=match_predictions(pc,tc,box_iou(tb,pb)) if len(preds) and len(tb) else torch.zeros((len(preds),10),dtype=torch.bool)
        metrics.stats['tp'].append(correct.numpy()); metrics.stats['conf'].append(conf.numpy()); metrics.stats['pred_cls'].append(pc.numpy()); metrics.stats['target_cls'].append(tc.numpy()); metrics.stats['target_img'].append(np.full(len(tc),image_index)); area=((tb[:,2]-tb[:,0])*(tb[:,3]-tb[:,1])).numpy(); sizes += [{'image':name,'class_id':int(c),'area_px':float(v)} for c,v in zip(tc,area)]
    metrics.process(save_dir=a.output.parent,plot=False); pr,re,m50,map95=metrics.mean_results(); per={}
    for c,n in ((0,'Flash point'),(1,'Big black spots')):
        cp,cr,ap50,ap=metrics.class_result(c); per[str(c)]={'name':n,'precision':float(cp),'recall':float(cr),'ap50':float(ap50),'ap50_95':float(ap)}
    out={'precision':float(pr),'recall':float(re),'map50':float(m50),'map50_95':float(map95),'map75':float(metrics.box.map75),'per_class':per,'size_diagnostic_rows':sizes,'development_status':'exploratory_outer_holdout'}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({k:v for k,v in out.items() if k!='size_diagnostic_rows'},indent=2))
if __name__=='__main__': main()
