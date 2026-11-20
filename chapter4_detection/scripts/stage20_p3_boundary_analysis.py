#!/usr/bin/env python3
"""Fold0-train-only P3 boundary-information audit for Stage20.

GT is used only to select an oracle diagnostic candidate and measure its error.
No holdout image or label is read and this script is not an inference method.
"""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from chapter4_detection.scripts.stage7_initialization_audit import real_batch


def xywh_to_xyxy(q):
    x, y, w, h = q
    return np.array([x-w/2, y-h/2, x+w/2, y+h/2], dtype=np.float64)


def box_iou_one_many(a, b):
    lt = np.maximum(a[:2], b[:, :2]); rb = np.minimum(a[2:], b[:, 2:])
    inter = np.maximum(rb-lt, 0).prod(1)
    aa = np.maximum(a[2:]-a[:2], 0).prod(); bb = np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)
    return inter / (aa+bb-inter+1e-12)


def sample(feat, x, y):
    """Bilinear sample at feature-map pixel coordinates (align_corners=False)."""
    h, w = feat.shape[-2:]
    grid = torch.tensor([[[[(x+.5)*2/w-1, (y+.5)*2/h-1]]]], dtype=feat.dtype)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0, 0]


def edge_vectors(feat, box, offset=.5):
    x1, y1, x2, y2 = box; cx=(x1+x2)/2; cy=(y1+y2)/2
    pts={'left':(x1,cy),'top':(cx,y1),'right':(x2,cy),'bottom':(cx,y2)}
    normals={'left':(1,0),'top':(0,1),'right':(-1,0),'bottom':(0,-1)}
    out={}; edges={}
    for name,(x,y) in pts.items():
        nx,ny=normals[name]; edges[name]=sample(feat,x,y)
        inside=sample(feat,x+nx*offset,y+ny*offset); outside=sample(feat,x-nx*offset,y-ny*offset)
        out[name]=inside-outside
    return sample(feat,cx,cy),edges,out


def cosine(a,b):
    return float(F.cosine_similarity(a.float(),b.float(),dim=0))


def ridge_group_cv(rows, center_only):
    """Group-held-out ridge probe; reports evidence, never tunes on holdout data."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        return {'available':False,'reason':str(e)}
    prefix='center_' if center_only else 'all_'
    X=np.stack([r[prefix+'features'] for r in rows]); Y=np.stack([r['targets'] for r in rows])
    groups=np.array([r['source_group'] for r in rows]); n=min(5,len(np.unique(groups)))
    if n<2:return {'available':False,'reason':'fewer than two source groups'}
    pred=np.zeros_like(Y); seen=np.zeros(len(rows),bool)
    for tr,te in GroupKFold(n_splits=n).split(X,Y,groups):
        sc=StandardScaler().fit(X[tr]); reg=Ridge(alpha=100.0).fit(sc.transform(X[tr]),Y[tr])
        pred[te]=reg.predict(sc.transform(X[te]));seen[te]=True
    mae=np.mean(np.abs(pred[seen]-Y[seen]),axis=0); base=np.mean(np.abs(Y[seen]-np.median(Y[seen],axis=0)),axis=0)
    return {'available':True,'n':int(seen.sum()),'groups':int(len(np.unique(groups))),
            'mae':dict(zip(('left','top','right','bottom'),map(float,mae))),
            'median_baseline_mae':dict(zip(('left','top','right','bottom'),map(float,base))),
            'mean_mae':float(mae.mean()),'mean_baseline_mae':float(base.mean())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path,required=True);p.add_argument('--train-list',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--imgsz',type=int,default=1536);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True); torch.manual_seed(42); np.random.seed(42)
    allowed={Path(x).resolve() for x in a.train_list.read_text().splitlines() if x.strip()}
    meta={}
    for r in csv.DictReader(a.gt_stats.open()):
        if r['split']=='train':meta[(r['image'],int(r['gt_id']))]=r
    model=YOLO(str(a.weights)).model.cpu().eval(); cap={}
    hook=model.model[-1].register_forward_pre_hook(lambda m,x:cap.update(p3=x[0][0].detach().cpu()))
    rows=[]; probe=[]
    for ni,image in enumerate(sorted(allowed)):
        label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
        batch=real_batch(image,label,a.imgsz,'cpu')
        with torch.no_grad(): decoded=model(batch['img'])[0][0].detach().cpu()
        feat=cap['p3']; boxes=decoded[:4].T.numpy(); scores=decoded[4:].T.numpy(); predxy=np.stack([xywh_to_xyxy(q) for q in boxes])
        H,W=feat.shape[-2:]; gt_boxes=batch['bboxes'].cpu().numpy(); classes=batch['cls'].flatten().long().cpu().numpy()
        for gid,(g,c) in enumerate(zip(gt_boxes,classes)):
            gt=xywh_to_xyxy(g*a.imgsz); vals=box_iou_one_many(gt,predxy); cls_scores=scores[:,c]
            valid=cls_scores>=1e-5
            if not valid.any():continue
            ids=np.where(valid)[0]; k=ids[np.argmax(vals[ids])]; pred=predxy[k]; piou=float(vals[k])
            # Input pixels -> P3 coordinates. Feature stride is measured rather than assumed.
            sx=W/a.imgsz; sy=H/a.imgsz; pb=pred*np.array([sx,sy,sx,sy]); gb=gt*np.array([sx,sy,sx,sy])
            center,edges,diffs=edge_vectors(feat,pb); bg=pb.copy(); shift=max(12.0,pb[2]-pb[0]+8.0)
            bg[[0,2]] += shift
            if bg[2]>=W-1:bg[[0,2]] -= 2*shift
            bg[0]=max(.5,bg[0]); bg[2]=min(W-1.5,bg[2]); bcenter,bedges,bdiffs=edge_vectors(feat,bg)
            target=np.array([pb[0]-gb[0],pb[1]-gb[1],pb[2]-gb[2],pb[3]-gb[3]],np.float64)
            edge_cat=torch.cat([diffs[x] for x in ('left','top','right','bottom')]).numpy()
            m=meta.get((image.name,gid),{}); rec={'image':image.name,'gt_id':gid,'class_id':int(c),'class_name':m.get('class_name',''),'source_group':m.get('source_group','UNKNOWN'),'oracle_iou':piou,'oracle_score':float(cls_scores[k]),'tiny_input':bool(min(g[2],g[3])*a.imgsz<8),'p3_box_width':float(pb[2]-pb[0]),'p3_box_height':float(pb[3]-pb[1])}
            for j,n in enumerate(('left','top','right','bottom')):
                rec[n+'_error_p3']=float(target[j]);rec[n+'_contrast_l2']=float(torch.linalg.vector_norm(diffs[n]));rec[n+'_background_contrast_l2']=float(torch.linalg.vector_norm(bdiffs[n]));rec[n+'_center_cosine']=cosine(center,edges[n])
            rec['opposite_lr_cosine']=cosine(edges['left'],edges['right']);rec['opposite_tb_cosine']=cosine(edges['top'],edges['bottom'])
            rows.append(rec);probe.append({'source_group':rec['source_group'],'targets':target,'center_features':center.numpy(),'all_features':np.concatenate([center.numpy(),edge_cat])})
        if (ni+1)%10==0:print(f'{ni+1}/{len(allowed)} images',flush=True)
    hook.remove()
    with (a.output/'p3_boundary_instances.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    def vals(key,sub=rows):return np.array([float(x[key]) for x in sub])
    summary={'scope':'Fold0 train only','weights':str(a.weights.resolve()),'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'imgsz':a.imgsz,'images':len(allowed),'instances':len(rows),'tiny_instances':sum(r['tiny_input'] for r in rows),'oracle_iou_median':float(np.median(vals('oracle_iou'))),'p3_box_size_median':[float(np.median(vals('p3_box_width'))),float(np.median(vals('p3_box_height')))]}
    summary['edges']={}
    for n in ('left','top','right','bottom'):
        x=vals(n+'_error_p3');q=vals(n+'_contrast_l2');b=vals(n+'_background_contrast_l2')
        summary['edges'][n]={'error_abs_median_p3':float(np.median(abs(x))),'defect_contrast_median':float(np.median(q)),'background_contrast_median':float(np.median(b)),'defect_vs_background_median_ratio':float(np.median(q)/(np.median(b)+1e-12)),'abs_error_contrast_correlation':float(np.corrcoef(abs(x),q)[0,1]),'center_edge_cosine_median':float(np.median(vals(n+'_center_cosine')))}
    summary['opposite_edge_cosine_median']={'left_right':float(np.median(vals('opposite_lr_cosine'))),'top_bottom':float(np.median(vals('opposite_tb_cosine')))}
    summary['subgroups']={}
    for name,sub in [('Flash',[r for r in rows if r['class_id']==0]),('Black',[r for r in rows if r['class_id']==1]),('tiny',[r for r in rows if r['tiny_input']])]:
        summary['subgroups'][name]={'n':len(sub),'oracle_iou_median':float(np.median(vals('oracle_iou',sub))),
            'edge_contrast_median':{e:float(np.median(vals(e+'_contrast_l2',sub))) for e in ('left','top','right','bottom')},
            'center_edge_cosine_median':{e:float(np.median(vals(e+'_center_cosine',sub))) for e in ('left','top','right','bottom')}}
    summary['group_cv_ridge']={'center_only':ridge_group_cv(probe,True),'center_plus_boundary':ridge_group_cv(probe,False)}
    c=summary['group_cv_ridge']; useful=bool(c['center_only'].get('available') and c['center_plus_boundary'].get('mean_mae',1e9)<.9*c['center_only'].get('mean_mae',0))
    ratios=[summary['edges'][n]['defect_vs_background_median_ratio'] for n in summary['edges']]
    summary['predeclared_gate']={'rule':'implement only if boundary probe reduces group-CV MAE by >=10% and median defect/background contrast ratio >1.10','probe_mae_improvement':None if not c['center_only'].get('available') else 1-c['center_plus_boundary']['mean_mae']/c['center_only']['mean_mae'],'median_contrast_ratio':float(np.median(ratios)),'pass':bool(useful and np.median(ratios)>1.10)}
    (a.output/'p3_boundary_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    try:
        import matplotlib.pyplot as plt
        names=('left','top','right','bottom'); fig,ax=plt.subplots(1,2,figsize=(10,4))
        ax[0].boxplot([[float(r[n+'_contrast_l2']) for r in rows] for n in names],labels=names);ax[0].set_title('Predicted-edge P3 contrast');ax[0].set_ylabel('L2 inside-outside')
        ax[1].boxplot([[float(r[n+'_center_cosine']) for r in rows] for n in names],labels=names);ax[1].set_title('Center / edge cosine');ax[1].set_ylim(-.1,1.05)
        fig.tight_layout();fig.savefig(a.output/'p3_boundary_information.png',dpi=180);plt.close(fig)
    except Exception as e: print('plot skipped',e)


if __name__=='__main__':main()
