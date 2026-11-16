#!/usr/bin/env python3
"""Paired train-only frequency analysis for Plastic_Bomo HBB candidate ROIs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


EPS = 1e-12


def overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def metrics(gray: np.ndarray) -> dict[str, float]:
    x = gray.astype(np.float32) / 255.0
    h, w = x.shape
    x = x[: h - h % 2, : w - w % 2]
    a, b, c, d = x[0::2, 0::2], x[0::2, 1::2], x[1::2, 0::2], x[1::2, 1::2]
    ll, lh, hl, hh = (a+b+c+d)/2, (-a-b+c+d)/2, (-a+b-c+d)/2, (a-b-c+d)/2
    energies = [float(np.mean(z*z)) for z in (ll, lh, hl, hh)]
    spec = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2
    yy, xx = np.indices(spec.shape); cy, cx = (np.asarray(spec.shape)-1)/2
    radius = np.sqrt(((yy-cy)/(max(1, spec.shape[0]/2)))**2 + ((xx-cx)/(max(1, spec.shape[1]/2)))**2)
    high = float(spec[radius >= .35].sum() / (spec.sum()+EPS))
    flat = spec[radius >= .08]; periodicity = float(flat.max()/(flat.mean()+EPS)) if flat.size else 0.0
    sx=cv2.Sobel(x,cv2.CV_32F,1,0,ksize=3);sy=cv2.Sobel(x,cv2.CV_32F,0,1,ksize=3)
    lap=cv2.Laplacian(x,cv2.CV_32F,ksize=3)
    hist=np.histogram(gray,bins=32,range=(0,256))[0].astype(float);p=hist/(hist.sum()+EPS);entropy=float(-(p[p>0]*np.log2(p[p>0])).sum())
    eh=sum(energies[1:]);total=sum(energies)
    return {"e_ll":energies[0],"e_lh":energies[1],"e_hl":energies[2],"e_hh":energies[3],"e_high":eh,"haar_high_ratio":eh/(total+EPS),"fft_high_ratio":high,"fft_periodicity":periodicity,"sobel_mean":float(np.hypot(sx,sy).mean()),"laplacian_abs_mean":float(np.abs(lap).mean()),"texture_entropy":entropy,"mean_intensity":float(x.mean()),"std_intensity":float(x.std())}


def find_background(gray, integral, box, all_boxes, rng):
    x1,y1,x2,y2=box;w=x2-x1;h=y2-y1;H,W=gray.shape
    candidates=[]
    for _ in range(500):
        bx1=int(rng.integers(0,max(1,W-w+1)));by1=int(rng.integers(0,max(1,H-h+1)));b=(bx1,by1,bx1+w,by1+h)
        if not any(overlap(b,g) for g in all_boxes):candidates.append(b)
    if not candidates:return None
    # Prefer similar local brightness so frequency comparisons are not merely luminance comparisons.
    target=float(gray[y1:y2,x1:x2].mean())
    def rect_mean(q):
        qx1,qy1,qx2,qy2=q
        return float((integral[qy2,qx2]-integral[qy1,qx2]-integral[qy2,qx1]+integral[qy1,qx1])/(w*h))
    return min(candidates,key=lambda b:abs(rect_mean(b)-target))


def bootstrap(rows, keys, n, seed):
    rng=np.random.default_rng(seed);groups=defaultdict(list)
    for r in rows:groups[r["source_group"]].append(r)
    names=list(groups);out={}
    for cls in ("all",0,1):
        subset=[r for r in rows if cls=="all" or r["class_id"]==cls];valid_groups=sorted({r["source_group"] for r in subset});by=defaultdict(list)
        for r in subset:by[r["source_group"]].append(r)
        out[str(cls)]={}
        for key in keys:
            observed=float(np.mean([r[f"target_{key}"]-r[f"background_{key}"] for r in subset]));samples=[]
            for _ in range(n):
                picked=rng.choice(valid_groups,len(valid_groups),replace=True);vals=[]
                for g in picked:vals.extend(r[f"target_{key}"]-r[f"background_{key}"] for r in by[g])
                samples.append(np.mean(vals))
            out[str(cls)][key]={"paired_difference_mean":observed,"ci95":[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],"source_groups":len(valid_groups),"rois":len(subset)}
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--bootstrap",type=int,default=2000);p.add_argument("--seed",type=int,default=42);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text());rng=np.random.default_rng(a.seed);rows=[];excluded=[]
    for rec in manifest["records"]:
        if rec["split"]!="train":continue
        im=cv2.imread(rec["image"]);gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);integral=cv2.integral(gray,sdepth=cv2.CV_64F);H,W=gray.shape
        labels=[z.split() for z in Path(rec["label"]).read_text().splitlines() if z.strip()];boxes=[]
        for z in labels:
            _,cx,cy,bw,bh=map(float,z);x1=max(0,int(round((cx-bw/2)*W)));y1=max(0,int(round((cy-bh/2)*H)));x2=min(W,int(round((cx+bw/2)*W)));y2=min(H,int(round((cy+bh/2)*H)));boxes.append((x1,y1,x2,y2))
        for index,(z,box) in enumerate(zip(labels,boxes)):
            cls=int(z[0]);x1,y1,x2,y2=box;w=x2-x1;h=y2-y1
            if min(w,h)<4:excluded.append({"image":rec["image_name"],"index":index,"reason":"roi_too_small"});continue
            # Context must be complete; it is used only for ring contrast, never as pixel-level defect truth.
            pad=max(2,int(round(.5*max(w,h))));cx1,cy1,cx2,cy2=x1-pad,y1-pad,x2+pad,y2+pad
            if cx1<0 or cy1<0 or cx2>W or cy2>H:excluded.append({"image":rec["image_name"],"index":index,"reason":"incomplete_context"});continue
            bg=find_background(gray,integral,box,boxes,rng)
            if bg is None:excluded.append({"image":rec["image_name"],"index":index,"reason":"no_nonoverlap_background"});continue
            target=gray[y1:y2,x1:x2];background=gray[bg[1]:bg[3],bg[0]:bg[2]];tm=metrics(target);bm=metrics(background)
            context=gray[cy1:cy2,cx1:cx2];mask=np.ones(context.shape,bool);mask[y1-cy1:y2-cy1,x1-cx1:x2-cx1]=False;ring=context[mask]
            row={"image":rec["image_name"],"canonical_image_id":rec["canonical_image_id"],"source_group":rec["source_group"],"class_id":cls,"roi_index":index,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"width":w,"height":h,"area":w*h,"background_x1":bg[0],"background_y1":bg[1],"local_contrast":abs(float(target.mean())-float(ring.mean()))/255.0,"ring_mean_intensity":float(ring.mean()/255.0)}
            for key,val in tm.items():row[f"target_{key}"]=val
            for key,val in bm.items():row[f"background_{key}"]=val
            rows.append(row)
    keys=list(metrics(np.zeros((8,8),np.uint8)))
    with (a.output_dir/"frequency_roi_statistics.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    boot=bootstrap(rows,keys,a.bootstrap,a.seed);summary={"protocol":"train_only_paired_ROI_source_group_bootstrap","manifest_protocol_sha256":manifest["protocol_sha256"],"included":len(rows),"excluded":excluded,"class_counts":{str(c):sum(r["class_id"]==c for r in rows) for c in (0,1)},"bootstrap":boot}
    (a.output_dir/"frequency_analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axs,("haar_high_ratio","fft_high_ratio","laplacian_abs_mean"),("Haar high ratio","FFT high ratio","Laplacian response")):
        data=[];labels=[]
        for c,name in ((0,"Flash"),(1,"Black")):
            q=[r for r in rows if r["class_id"]==c];data.extend([[r[f"target_{key}"] for r in q],[r[f"background_{key}"] for r in q]]);labels.extend([f"{name}\nROI",f"{name}\nBG"])
        ax.boxplot(data,labels=labels,showfliers=False);ax.set_title(title)
    fig.tight_layout();fig.savefig(a.output_dir/"frequency_distributions.png",dpi=180);plt.close(fig)
    print(json.dumps({"included":len(rows),"excluded":len(excluded),"class_counts":summary["class_counts"]},indent=2))


if __name__=="__main__":main()
