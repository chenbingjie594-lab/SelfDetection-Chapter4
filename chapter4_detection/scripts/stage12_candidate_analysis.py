#!/usr/bin/env python3
"""Trace YOLO candidate selection and analyze score/localization alignment on frozen OOF caches."""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, pearsonr, spearmanr
from ultralytics.utils.nms import TorchNMS, non_max_suppression
from ultralytics.utils.ops import xywh2xyxy

from chapter4_detection.scripts.evaluate_original import load_gt


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    lt=np.maximum(a[:,None,:2],b[None,:,:2]); rb=np.minimum(a[:,None,2:],b[None,:,2:])
    wh=np.maximum(rb-lt,0); inter=wh[...,0]*wh[...,1]
    aa=np.maximum(a[:,2]-a[:,0],0)*np.maximum(a[:,3]-a[:,1],0)
    bb=np.maximum(b[:,2]-b[:,0],0)*np.maximum(b[:,3]-b[:,1],0)
    return inter/(aa[:,None]+bb[None,:]-inter+1e-12)


def traceable_nms(prediction: torch.Tensor, conf=.001, iou=.7, max_det=300, max_nms=30000, max_wh=7680):
    """Match Ultralytics multi-label class-aware NMS while retaining candidate/class indices and reasons."""
    assert prediction.ndim==3 and prediction.shape[0]==1
    nc=prediction.shape[1]-4; raw=prediction[0].T.clone(); boxes=xywh2xyxy(raw[:,:4]); scores=raw[:,4:]
    states=[]
    for anchor in range(len(raw)):
        mx=float(scores[anchor].max())
        if mx<=conf:
            states.append({'anchor_id':anchor,'class_id':-1,'score':mx,'reason':'below_preconf'})
    ii,jj=torch.where(scores>conf)
    det=torch.cat((boxes[ii],scores[ii,jj,None],jj[:,None].float()),1)
    ids=torch.stack((ii,jj),1)
    order=torch.arange(len(det),device=det.device)
    if len(det)>max_nms:
        keep=det[:,4].argsort(descending=True)[:max_nms]; dropped=set(order.tolist())-set(keep.tolist())
        for k in dropped: states.append({'anchor_id':int(ids[k,0]),'class_id':int(ids[k,1]),'score':float(det[k,4]),'reason':'max_nms'})
        det,ids,order=det[keep],ids[keep],order[keep]
    shifted=det[:,:4]+det[:,5:6]*max_wh
    keep=TorchNMS.nms(shifted,det[:,4],iou); nms_keep=keep[:max_det]
    nms_set=set(keep.tolist()); final_set=set(nms_keep.tolist())
    for k in range(len(det)):
        rec={'anchor_id':int(ids[k,0]),'class_id':int(ids[k,1]),'score':float(det[k,4])}
        if k in final_set: rec['reason']='kept'
        elif k in nms_set: rec['reason']='max_det'
        else:
            rec['reason']='nms_suppressed'; higher=nms_keep[det[nms_keep,5]==det[k,5]]
            if len(higher):
                ov=box_iou(det[k:k+1,:4].cpu().numpy(),det[higher,:4].cpu().numpy())[0]
                h=int(higher[int(np.argmax(ov))]); rec.update(suppressor_anchor_id=int(ids[h,0]),suppressor_score=float(det[h,4]),suppressor_overlap=float(ov.max()))
        states.append(rec)
    return det[nms_keep], ids[nms_keep], states


def scale_boxes(boxes, rec):
    q=boxes.copy(); gain=rec['ratio_pad'][0]; pad=rec['ratio_pad'][1]
    q[:,[0,2]]=(q[:,[0,2]]-pad[0])/gain[1]; q[:,[1,3]]=(q[:,[1,3]]-pad[1])/gain[0]
    # DetectionValidator scales back to the native image but does not clip the
    # serialized boxes in this installed version; preserve that exact behavior.
    return q


def summarize_corr(rows):
    out={}
    for key,sel in [('all',rows),('flash',[r for r in rows if r['class_id']==0]),('black',[r for r in rows if r['class_id']==1])]+[(p,[r for r in rows if r['level']==p]) for p in ('P3','P4','P5')]:
        # one top-scoring candidate per image/class reduces pseudo-replication
        best={}
        for r in sel:
            k=(r['image'],r['class_id']);
            if k not in best or r['score']>best[k]['score']:best[k]=r
        z=list(best.values()); x=np.array([r['score'] for r in z]); y=np.array([r['quality_iou'] for r in z])
        out[key]={'n_image_class':len(z),'pearson':float(pearsonr(x,y).statistic) if len(z)>2 else None,'spearman':float(spearmanr(x,y).statistic) if len(z)>2 else None,'kendall':float(kendalltau(x,y).statistic) if len(z)>2 else None,'high_score_low_iou':int(sum(r['score']>=.25 and r['quality_iou']<.5 for r in z)),'low_score_high_iou':int(sum(r['score']<.25 and r['quality_iou']>=.75 for r in z))}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    trace_rows=[]; gt_rows=[]; consistency=[]; intervention=defaultdict(lambda:defaultdict(list)); raw_hashes={}
    gz_path=a.output_dir/'candidate_trace.jsonl.gz'
    with gzip.open(gz_path,'wt',encoding='utf8') as gz:
      for model in ('e0','cbcd'):
       for fold in range(3):
        root=a.raw_root/f'{model}_fold{fold}'; man=json.loads((root/'manifest.json').read_text())
        counts=[q[2]*q[3] for q in next(iter(man['records'].values()))['feature_shapes']]; levels=np.concatenate([np.full(n,f'P{i}') for n,i in zip(counts,(3,4,5))])
        for name,rec in man['records'].items():
         f=root/rec['file']; raw_hashes[str(f.relative_to(a.raw_root))]=hashlib.sha256(f.read_bytes()).hexdigest(); z=np.load(f); dec=torch.from_numpy(z['decoded'])[None]
         official,oidx=non_max_suppression(dec.clone(),.001,.7,nc=2,multi_label=True,max_det=300,return_idxs=True); traced,tids,states=traceable_nms(dec.clone())
         ok=len(official[0])==len(traced) and torch.allclose(official[0],traced,atol=1e-6,rtol=1e-6) and torch.equal(oidx[0],tids[:,0])
         consistency.append({'model':model,'fold':fold,'image':name,'official_n':len(official[0]),'trace_n':len(traced),'exact':bool(ok),'max_abs':float((official[0]-traced).abs().max()) if len(traced)==len(official[0]) and len(traced) else 0.})
         if not ok: raise RuntimeError(f'TraceableNMS mismatch: {model}/{fold}/{name}')
         H,W=rec['original_shape']; tc,tb=load_gt(a.labels/f'{Path(name).stem}.txt',W,H); gtc=tc.numpy().astype(int); gtb=tb.numpy()
         xywh=z['decoded'][:4].T; inp=np.column_stack((xywh[:,0]-xywh[:,2]/2,xywh[:,1]-xywh[:,3]/2,xywh[:,0]+xywh[:,2]/2,xywh[:,1]+xywh[:,3]/2)); orig=scale_boxes(inp,rec); sc=z['decoded'][4:].T
         state_map={(s['anchor_id'],s['class_id']):s for s in states if s['class_id']>=0}
         eligible=np.argwhere(sc>.001)
         local=[]
         for anchor,cls in eligible:
          same=gtb[gtc==cls]; quality=float(box_iou(orig[anchor:anchor+1],same).max()) if len(same) else 0.; st=state_map[(int(anchor),int(cls))]
          row={'model':model,'fold':fold,'image':name,'anchor_id':int(anchor),'candidate_id':f'{model}:{fold}:{name}:{anchor}:{cls}','class_id':int(cls),'level':str(levels[anchor]),'x1':float(orig[anchor,0]),'y1':float(orig[anchor,1]),'x2':float(orig[anchor,2]),'y2':float(orig[anchor,3]),'flash_score':float(sc[anchor,0]),'black_score':float(sc[anchor,1]),'score':float(sc[anchor,cls]),'quality_iou':quality,**{k:v for k,v in st.items() if k not in ('anchor_id','class_id','score')}}
          local.append(row);trace_rows.append(row);gz.write(json.dumps(row,separators=(',',':'))+'\n')
         for gi,(cls,box) in enumerate(zip(gtc,gtb)):
          cand=[r for r in local if r['class_id']==cls]; spatial=[r for r in cand if box_iou(np.array([[r['x1'],r['y1'],r['x2'],r['y2']]]),box[None])[0,0]>=.10]
          if not spatial: spatial=cand
          top=max(spatial,key=lambda r:r['score']) if spatial else None; oracle=max(spatial,key=lambda r:r['quality_iou']) if spatial else None
          if oracle is None or oracle['quality_iou']<.75:cat='A_generation_insufficient'
          elif top['quality_iou']<.75:cat='B_score_misranking'
          elif oracle['reason']=='nms_suppressed':cat='C_nms_suppression'
          elif oracle['reason'] in ('max_det','max_nms'):cat='D_other_postprocess'
          else:cat='E_correctly_retained'
          gt_rows.append({'model':model,'fold':fold,'image':name,'gt_id':gi,'class_id':int(cls),'gt_width':float(box[2]-box[0]),'gt_height':float(box[3]-box[1]),'category':cat,'top_score_iou':top['quality_iou'] if top else 0.,'oracle_iou':oracle['quality_iou'] if oracle else 0.,'delta_iou':(oracle['quality_iou']-top['quality_iou']) if oracle and top else 0.,'oracle_score_rank':sorted(spatial,key=lambda r:r['score'],reverse=True).index(oracle)+1 if oracle else None,'oracle_reason':oracle['reason'] if oracle else 'none','oracle_score':oracle['score'] if oracle else 0.,'top_score':top['score'] if top else 0.})
         for thr in (.5,.7,.9):
          out,_,_=traceable_nms(dec.clone(),iou=thr); b=scale_boxes(out[:,:4].numpy(),rec); intervention[model][str(thr)].append({'image':name,'width':W,'height':H,'predictions':[{'class_id':int(q[5]),'confidence':float(q[4]),'xyxy':[float(v) for v in bb]} for q,bb in zip(out.numpy(),b)]})
    # Compact tables; full eligible trace remains gzip.
    keys=sorted({k for r in trace_rows for k in r});
    with (a.output_dir/'candidate_trace.csv').open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(trace_rows)
    keys=list(gt_rows[0]);
    with (a.output_dir/'gt_failure_decomposition.csv').open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(gt_rows)
    summary={'consistency':{'images':len(consistency),'all_exact':all(r['exact'] for r in consistency),'max_abs':max(r['max_abs'] for r in consistency)},'correlations':{m:summarize_corr([r for r in trace_rows if r['model']==m]) for m in ('e0','cbcd')},'decomposition':{m:dict(Counter(r['category'] for r in gt_rows if r['model']==m)) for m in ('e0','cbcd')},'trace_rows':len(trace_rows),'raw_cache_sha256':raw_hashes}
    (a.output_dir/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output_dir/'nms_intervention_predictions.json').write_text(json.dumps(intervention)+'\n')
    (a.output_dir/'trace_consistency.json').write_text(json.dumps(consistency,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='raw_cache_sha256'},indent=2))

if __name__=='__main__':main()
