#!/usr/bin/env python3
"""Paired source-group bootstrap from frozen per-image predictions."""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from ultralytics.utils.metrics import DetMetrics, box_iou, compute_ap
from chapter4_detection.scripts.evaluate_original import load_gt, match_predictions

MODELS = {"E0":"e0", "E1":"e1", "E3":"e3", "A1":"e3_no_reference", "A2":"e3_no_gate"}
SEEDS = (42,3407,2026)
PAIRS = (("E3","E0"),("E3","E1"),("E3","A1"),("E3","A2"),("A1","E0"))

def image_stats(record, label):
    tc,tb=load_gt(label,record["width"],record["height"]); ps=record["predictions"]
    pc=torch.tensor([p["class_id"] for p in ps],dtype=torch.float32)
    pb=torch.tensor([p["xyxy"] for p in ps],dtype=torch.float32).reshape(-1,4)
    cf=torch.tensor([p["confidence"] for p in ps],dtype=torch.float32)
    cr=match_predictions(pc,tc,box_iou(tb,pb)) if len(ps) and len(tb) else torch.zeros((len(ps),10),dtype=torch.bool)
    return cr.numpy(),cf.numpy(),pc.numpy(),tc.numpy()

def evaluate(items):
    m=DetMetrics(names={0:"Flash point",1:"Big black spots"})
    for i,(tp,cf,pc,tc) in enumerate(items):
        m.stats["tp"].append(tp); m.stats["conf"].append(cf); m.stats["pred_cls"].append(pc)
        m.stats["target_cls"].append(tc); m.stats["target_img"].append(np.full(len(tc),i))
    m.process(plot=False)
    return np.array([m.box.map,m.box.map75,m.maps[0],m.maps[1]],dtype=float)

def prepare_weighted(per_image, image_group):
    """Pre-sort predictions once; bootstrap then changes only group weights."""
    out={}
    for cls in (0,1):
        conf=[]; tp=[]; pg=[]; gt=defaultdict(int)
        for name,(itp,icf,ipc,itc) in per_image.items():
            g=image_group[name]; gt[g]+=int(np.sum(itc==cls)); mask=ipc==cls
            conf.extend(icf[mask]); tp.extend(itp[mask]); pg.extend([g]*int(mask.sum()))
        order=np.argsort(-np.asarray(conf)); out[cls]=(np.asarray(tp,bool)[order],np.asarray(pg,object)[order],dict(gt))
    return out

def weighted_evaluate(prepared, multiplicity):
    aps=np.zeros((2,10),float)
    for cls in (0,1):
        tp,groups,gt=prepared[cls]; weights=np.asarray([multiplicity.get(g,0) for g in groups],float)
        keep=weights>0; weights=weights[keep]; z=tp[keep]
        n_gt=sum(multiplicity.get(g,0)*n for g,n in gt.items())
        if not n_gt: continue
        for j in range(10):
            tpc=np.cumsum(weights*z[:,j]); fpc=np.cumsum(weights*(~z[:,j])); recall=tpc/n_gt; precision=tpc/(tpc+fpc+1e-16)
            aps[cls,j]=compute_ap(recall,precision)[0] if len(recall) else 0.0
    return np.array([aps.mean(),aps[:,5].mean(),aps[0].mean(),aps[1].mean()])

def main():
    a=argparse.ArgumentParser(); a.add_argument("--results",type=Path,default=Path("chapter4_detection/results")); a.add_argument("--manifest",type=Path,required=True); a.add_argument("--labels",type=Path,required=True); a.add_argument("--output-dir",type=Path,required=True); a.add_argument("--repeats",type=int,default=5000); a.add_argument("--seed",type=int,default=20261115); x=a.parse_args()
    recs=json.loads(x.manifest.read_text())["records"]; val=[r for r in recs if r["split"]=="val"]
    groups=defaultdict(list)
    for r in val: groups[r["source_group"]].append(r["image_name"])
    gnames=sorted(groups); cache={}; original={}
    for model,prefix in MODELS.items():
      for seed in SEEDS:
        p=x.results/f"{prefix}_seed{seed}_validator_predictions.json"; data=json.loads(p.read_text())
        cache[model,seed]={n:image_stats(data["images"][n],x.labels/f"{Path(n).stem}.txt") for n in data["images"]}
        original[model,seed]=evaluate(list(cache[model,seed].values()))
    image_group={n:g for g,names in groups.items() for n in names}; prepared={(m,s):prepare_weighted(cache[m,s],image_group) for m in MODELS for s in SEEDS}
    rng=np.random.default_rng(x.seed); rows=[]; missing={"flash":0,"black":0}
    for rep in range(x.repeats):
      sample=rng.choice(gnames,size=len(gnames),replace=True)
      names=[n for g in sample for n in groups[g]]; multiplicity={g:int(np.sum(sample==g)) for g in set(sample)}
      # Predeclared rule: retain replicate; class AP becomes 0 if absent.
      base_tc=np.concatenate([cache["E0",42][n][3] for n in names])
      missing["flash"]+=int(not np.any(base_tc==0)); missing["black"]+=int(not np.any(base_tc==1))
      vals={(m,s):weighted_evaluate(prepared[m,s],multiplicity) for m in MODELS for s in SEEDS}
      for left,right in PAIRS:
        diffs=np.stack([vals[left,s]-vals[right,s] for s in SEEDS])
        for si,s in enumerate(SEEDS): rows.append([rep,left,right,str(s),*diffs[si]])
        rows.append([rep,left,right,"seed_mean",*diffs.mean(0)])
    x.output_dir.mkdir(parents=True,exist_ok=True); metrics=("map50_95","ap75","flash_ap50_95","black_ap50_95")
    csvp=x.output_dir/"paired_group_bootstrap_distribution.csv"
    with csvp.open("w",newline="") as f:
      w=csv.writer(f,lineterminator="\n"); w.writerow(["replicate","left","right","seed",*metrics]); w.writerows(rows)
    arr=np.asarray([r[4:] for r in rows],float); summary={"protocol":{"unit":"source_group","groups":len(gnames),"images":len(val),"repeats":x.repeats,"seed":x.seed,"missing_class_rule":"retain; absent class AP=0","missing_class_replicates":missing},"original":{},"intervals":{}}
    for m in MODELS:
      z=np.stack([original[m,s] for s in SEEDS]); summary["original"][m]={"per_seed":{str(s):dict(zip(metrics,original[m,s])) for s in SEEDS},"mean":dict(zip(metrics,z.mean(0))),"sd":dict(zip(metrics,z.std(0,ddof=1)))}
    for l,r in PAIRS:
      summary["intervals"][f"{l}-{r}"]={}
      for seed in (*SEEDS,"seed_mean"):
        z=np.array([row[4:] for row in rows if row[1]==l and row[2]==r and row[3]==str(seed)])
        summary["intervals"][f"{l}-{r}"][str(seed)]={k:{"mean":float(z[:,i].mean()),"ci95":[float(v) for v in np.quantile(z[:,i],[.025,.975])],"p_gt_0":float((z[:,i]>0).mean())} for i,k in enumerate(metrics)}
    (x.output_dir/"paired_group_bootstrap.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary["protocol"],indent=2))
if __name__=="__main__": main()
