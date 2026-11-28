#!/usr/bin/env python3
"""Leakage-controlled per-source Platt calibration on an inner calibration split."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import torch

def label_path(image): return Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); inter=ix*iy
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9)
def ground_truth(image,w,h):
    rows=[]
    for z in label_path(image).read_text().splitlines():
        c,x,y,bw,bh=map(float,z.split()); rows.append((int(c),[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
    return rows
def annotate(payload, paths, source):
    out=[]
    for image in paths:
        item=payload['images'][Path(image).name]; gt=ground_truth(image,item['width'],item['height']); used=set()
        preds=sorted(item['predictions'],key=lambda q:q['confidence'],reverse=True)
        for q in preds:
            same=[(iou(q['xyxy'],b),j) for j,(c,b) in enumerate(gt) if c==int(q['class_id'])]; any_iou=max([iou(q['xyxy'],b) for _,b in gt] or [0]); best=max(same,default=(0,-1))
            if best[0]>=.5 and best[1] not in used: label=1; used.add(best[1]); reason='tp'
            elif any_iou<.1 or best[1] in used or (best[0]<.5 and any_iou>=.5): label=0; reason='background_or_duplicate_or_class_error'
            else: label=-1; reason='ignore_ambiguous_iou'
            out.append({'image':Path(image).name,'source':source,'class_id':int(q['class_id']),'score':float(q['confidence']),'label':label,'reason':reason,'max_same_class_iou':best[0],'max_any_iou':any_iou})
    return out
def fit_platt(rows):
    usable=[r for r in rows if r['label']>=0]; y=torch.tensor([r['label'] for r in usable],dtype=torch.float64); s=torch.tensor([r['score'] for r in usable],dtype=torch.float64).clamp(1e-6,1-1e-6); x=torch.logit(s)
    if len(y)==0 or y.sum()==0 or y.sum()==len(y): return {'a':1.,'b':0.,'n':len(y),'positive':int(y.sum()),'status':'insufficient'}
    a=torch.tensor(1.,dtype=torch.float64,requires_grad=True); b=torch.tensor(0.,dtype=torch.float64,requires_grad=True); opt=torch.optim.LBFGS([a,b],lr=.2,max_iter=200,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss=torch.nn.functional.binary_cross_entropy_with_logits(a*x+b,y); loss.backward(); return loss
    opt.step(closure); return {'a':float(a.detach()),'b':float(b.detach()),'n':len(y),'positive':int(y.sum()),'status':'fit'}
def apply(score,p):
    z=p['a']*math.log(min(max(score,1e-6),1-1e-6)/(1-min(max(score,1e-6),1-1e-6)))+p['b']; return 1/(1+math.exp(-max(-50,min(50,z))))
def diagnostics(rows,params):
    u=[r for r in rows if r['label']>=0]; p=np.array([apply(r['score'],params[str(r['class_id'])]) for r in u]); y=np.array([r['label'] for r in u]); ece=0.
    for lo in np.linspace(0,1,11)[:-1]:
        m=(p>=lo)&(p<lo+.1)
        if m.any(): ece+=m.mean()*abs(p[m].mean()-y[m].mean())
    return {'n':len(u),'positive':int(y.sum()),'brier':float(np.mean((p-y)**2)),'ece10':float(ece)}
def thresholds(rows,params,nimages):
    scored=sorted([(apply(r['score'],params[str(r['class_id'])]),r) for r in rows if r['label']>=0],reverse=True,key=lambda z:z[0]); result={}
    for target in (.1,.25,.5):
        fp_limit=target*nimages; best=(1.,0,0)
        tp=fp=0
        for score,r in scored:
            tp+=r['label']==1; fp+=r['label']==0
            if fp<=fp_limit: best=(score,tp,fp)
        result[str(target)]={'threshold':best[0],'tp':best[1],'fp':best[2],'fppi':best[2]/nimages}
    return result
def main():
    p=argparse.ArgumentParser(); p.add_argument('--images-list',type=Path,required=True); p.add_argument('--e0-predictions',type=Path,required=True); p.add_argument('--fspr-predictions',type=Path,required=True); p.add_argument('--experiment',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); paths=[x for x in a.images_list.read_text().splitlines() if x.strip()]
    e0=json.loads(a.e0_predictions.read_text()); fspr=json.loads(a.fspr_predictions.read_text()); rows=annotate(e0,paths,'E0')+annotate(fspr,paths,'FSPR'); result={'experiment':a.experiment,'images':len(paths),'split':'inner-calibration','outer_holdout_used':False,'frozen_e0_seen_inner_calibration':True,'sources':{}}
    for source in ('E0','FSPR'):
        sr=[r for r in rows if r['source']==source]; shared=fit_platt(sr); params={}
        for c in (0,1):
            cr=[r for r in sr if r['class_id']==c]; candidate=fit_platt(cr); params[str(c)]=candidate if candidate['positive']>=10 else dict(shared,fallback='shared_insufficient_class_positives')
        result['sources'][source]={'shared':shared,'classes':params,'diagnostics':diagnostics(sr,params),'thresholds':thresholds(sr,params,len(paths)),'counts':{'tp':sum(r['label']==1 for r in sr),'fp':sum(r['label']==0 for r in sr),'ignore':sum(r['label']<0 for r in sr)}}
    result['rows_sha256']=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); a.output.with_name(a.output.stem+'_rows.json').write_text(json.dumps(rows)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
