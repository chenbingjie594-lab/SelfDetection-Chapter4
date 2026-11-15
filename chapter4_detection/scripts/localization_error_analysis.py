#!/usr/bin/env python3
"""Unified localization and error taxonomy analysis for frozen predictions."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2, numpy as np
from chapter4_detection.scripts.evaluate_original import load_gt
from chapter4_detection.scripts.slicing import box_iou

MODELS={"E0":"e0","E3":"e3","A1":"e3_no_reference"}; SEEDS=(42,3407,2026)
def one_to_one(preds,gc,gb,conf=.001,class_aware=True):
 order=sorted(range(len(preds)),key=lambda i:preds[i]["confidence"],reverse=True); used=set(); matches=[]
 for pi in order:
  p=preds[pi]; cand=[i for i in range(len(gb)) if i not in used and (not class_aware or gc[i]==p["class_id"])]
  if not cand: continue
  ious=box_iou(np.asarray(p["xyxy"]),gb[cand]); j=int(np.argmax(ious)); gi=cand[j]
  if ious[j]>=conf: used.add(gi); matches.append((pi,gi,float(ious[j])))
 return matches
def main():
 p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); p.add_argument("--labels",type=Path,required=True); p.add_argument("--images",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--thresholds",type=Path); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
 low=set()
 if a.thresholds and a.thresholds.exists():
  d=json.loads(a.thresholds.read_text()); low={(z["image"],int(z["gt_index"])) for z in d.get("low_contrast_black_targets",d.get("black_low_contrast_targets",[]))}
 rows=[]; detail=[]; loaded={}
 for model,prefix in MODELS.items():
  for seed in SEEDS:
   data=json.loads((a.results/f"{prefix}_seed{seed}_validator_predictions.json").read_text()); loaded[model,seed]=data
   accum={"n_gt":0,"iou50":0,"iou75":0,"iou90":0,"center_error":[],"width_rel_error":[],"height_rel_error":[]}
   errors={str(t):{"0":{"tp":0,"fp":0,"fn":0,"background_fp":0,"class_confusion":0,"localization":0,"duplicate":0},"1":{"tp":0,"fp":0,"fn":0,"background_fp":0,"class_confusion":0,"localization":0,"duplicate":0}} for t in (.10,.25)}
   for name,rec in data["images"].items():
    gc,gbt=load_gt(a.labels/f"{Path(name).stem}.txt",rec["width"],rec["height"]); gc=gc.numpy().astype(int); gb=gbt.numpy(); preds=rec["predictions"]
    mat=one_to_one(preds,gc,gb,0,class_aware=True); bygt={g:(pi,iou) for pi,g,iou in mat}
    for gi,b in enumerate(gb):
     accum["n_gt"]+=1
     if gi in bygt:
      pi,iou=bygt[gi]; pb=np.asarray(preds[pi]["xyxy"]); gw,gh=b[2]-b[0],b[3]-b[1]; pw,ph=pb[2]-pb[0],pb[3]-pb[1]
      accum["iou50"]+=iou>=.5; accum["iou75"]+=iou>=.75; accum["iou90"]+=iou>=.9
      if iou < .5: continue
      accum["center_error"].append(float(np.linalg.norm([(pb[0]+pb[2]-b[0]-b[2])/2,(pb[1]+pb[3]-b[1]-b[3])/2])))
      accum["width_rel_error"].append(float(abs(pw-gw)/gw)); accum["height_rel_error"].append(float(abs(ph-gh)/gh))
      detail.append({"model":model,"seed":seed,"image":name,"gt_index":gi,"class_id":int(gc[gi]),"iou":iou,"center_error":accum["center_error"][-1],"width_rel_error":accum["width_rel_error"][-1],"height_rel_error":accum["height_rel_error"][-1],"low_contrast_black":(name,gi) in low})
    for t in (.10,.25):
     sel=[q for q in preds if q["confidence"]>=t]
     for c in (0,1):
      idx=[i for i,q in enumerate(sel) if q["class_id"]==c]; used=set(); tp=0
      for pi in sorted(idx,key=lambda i:sel[i]["confidence"],reverse=True):
       ious=box_iou(np.asarray(sel[pi]["xyxy"]),gb) if len(gb) else np.array([]); same=[j for j in range(len(gb)) if gc[j]==c and j not in used]
       if same and ious[same].max()>=.5: used.add(same[int(np.argmax(ious[same]))]);tp+=1;continue
       e=errors[str(t)][str(c)]; best=float(ious.max()) if len(ious) else 0; bj=int(ious.argmax()) if len(ious) else -1
       if best>=.5 and gc[bj]!=c:e["class_confusion"]+=1
       elif best>=.1:e["localization"]+=1
       elif any(gc[j]==c and ious[j]>=.5 for j in range(len(gb))):e["duplicate"]+=1
       else:e["background_fp"]+=1
      e=errors[str(t)][str(c)];e["tp"]+=tp;e["fp"]+=len(idx)-tp;e["fn"]+=int((gc==c).sum())-tp
   rows.append({"model":model,"seed":seed,"n_gt":accum["n_gt"],"iou50_rate":accum["iou50"]/accum["n_gt"],"iou75_rate":accum["iou75"]/accum["n_gt"],"iou90_rate":accum["iou90"]/accum["n_gt"],"center_error_mean":np.mean(accum["center_error"]),"width_rel_error_mean":np.mean(accum["width_rel_error"]),"height_rel_error_mean":np.mean(accum["height_rel_error"]),"errors":errors})
 (a.output_dir/"localization_error_analysis.json").write_text(json.dumps({"summary":rows,"instances":detail},indent=2)+"\n")
 with (a.output_dir/"localization_per_instance.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=detail[0].keys(),lineterminator="\n");w.writeheader();w.writerows(detail)
 # deterministic examples: largest E3-vs-E0 IoU improvement, seed42
 by={(z["model"],z["seed"],z["image"],z["gt_index"]):z for z in detail}; candidates=[]
 for k,z in by.items():
  if k[0]=="E3" and k[1]==42 and ("E0",42,k[2],k[3]) in by:candidates.append((z["iou"]-by[("E0",42,k[2],k[3])]["iou"],k[2]))
 vis=a.output_dir/"localization_visualizations";vis.mkdir(exist_ok=True)
 for stale in vis.glob("*.jpg"): stale.unlink()
 for rank,(_,name) in enumerate(sorted(candidates,reverse=True)[:5]):
  im=cv2.imread(str(a.images/name)); panels=[]
  for m in ("E0","E3","A1"):
   x=im.copy(); rec=loaded[m,42]["images"][name]; gc,gb=load_gt(a.labels/f"{Path(name).stem}.txt",rec["width"],rec["height"])
   for b in gb.numpy().astype(int):cv2.rectangle(x,tuple(b[:2]),tuple(b[2:]),(0,255,0),3)
   for q in rec["predictions"]:
    if q["confidence"]>=.10:
     b=np.array(q["xyxy"],int);cv2.rectangle(x,tuple(b[:2]),tuple(b[2:]),(0,0,255),2)
   cv2.putText(x,m,(30,60),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),4);panels.append(cv2.resize(x,(1024,256)))
  cv2.imwrite(str(vis/f"{rank}_{Path(name).stem}.jpg"),np.concatenate(panels,axis=1))
 print(json.dumps({"summaries":len(rows),"instances":len(detail),"visualizations":min(5,len(candidates))}))
if __name__=="__main__":main()
