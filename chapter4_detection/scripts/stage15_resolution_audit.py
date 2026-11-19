#!/usr/bin/env python3
"""Frozen original-vs-down/up sampled ROI information audit for Stage15."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, numpy as np, pandas as pd
from skimage.metrics import structural_similarity
from chapter4_detection.scripts.slicing import yolo_to_xyxy

def responses(gray: np.ndarray) -> dict[str,float]:
    f=gray.astype(np.float32)
    gx=cv2.Sobel(f,cv2.CV_32F,1,0,ksize=3);gy=cv2.Sobel(f,cv2.CV_32F,0,1,ksize=3)
    lap=cv2.Laplacian(f,cv2.CV_32F,ksize=3)
    mag=np.hypot(gx,gy)
    return {'gradient_mean':float(mag.mean()),'gradient_peak':float(np.quantile(mag,.95)),
            'horizontal_edge_mean':float(np.abs(gx).mean()),'vertical_edge_mean':float(np.abs(gy).mean()),
            'laplacian_abs_mean':float(np.abs(lap).mean()),'intensity_std':float(f.std())}

def crop_box(cx,cy,w,h,W,H):
    w=min(int(w),W);h=min(int(h),H);x1=int(np.clip(round(cx-w/2),0,W-w));y1=int(np.clip(round(cy-h/2),0,H-h));return x1,y1,x1+w,y1+h

def nonoverlap_background(gt,W,H,rw,rh):
    x1,y1,x2,y2=gt;cx=(x1+x2)/2;cy=(y1+y2)/2
    # Frozen deterministic order: vertical then horizontal offsets, far enough not to intersect GT.
    for dx,dy in [(0,-2*rh),(0,2*rh),(-2*rw,0),(2*rw,0)]:
        b=crop_box(cx+dx,cy+dy,rw,rh,W,H)
        if min(b[2],x2)<=max(b[0],x1) or min(b[3],y2)<=max(b[1],y1):return b
    raise RuntimeError('unable to construct background ROI')

def quality(a,b):
    mse=float(np.mean((a.astype(np.float32)-b.astype(np.float32))**2));psnr=float('inf') if mse==0 else 10*np.log10(255**2/mse)
    win=min(7,a.shape[0]//2*2-1,a.shape[1]//2*2-1)
    ssim=float(structural_similarity(a,b,data_range=255,channel_axis=2,win_size=max(3,win)))
    return psnr,ssim

def edge_peak_shift(a,b):
    def peak(x):
        g=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY).astype(np.float32);sx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3);sy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3);return np.unravel_index(np.argmax(np.hypot(sx,sy)),g.shape)
    p,q=peak(a),peak(b);return float(np.hypot(p[0]-q[0],p[1]-q[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(a.manifest.read_text());groups={r['image_name']:r for r in manifest['records']};rows=[];examples=[]
    for ip in sorted((a.dataset/'images/train').glob('*')):
        im=cv2.imread(str(ip));H,W=im.shape[:2]
        if (W,H)!=(4096,1024):raise RuntimeError(f'unexpected shape {ip}: {(W,H)}')
        # Frozen OpenCV INTER_AREA downsample and INTER_LINEAR upsample; integer endpoints, no align_corners concept.
        low=cv2.resize(im,(1536,384),interpolation=cv2.INTER_AREA);rec=cv2.resize(low,(4096,1024),interpolation=cv2.INTER_LINEAR)
        lab=a.dataset/'labels/train'/f'{ip.stem}.txt';items=[x.split() for x in lab.read_text().splitlines() if x.strip()]
        boxes=yolo_to_xyxy(np.array([[float(v) for v in x[1:]] for x in items]),W,H)
        for gi,(it,gt) in enumerate(zip(items,boxes)):
            x1,y1,x2,y2=gt;gw=x2-x1;gh=y2-y1;rw=max(32,int(np.ceil(gw*3)));rh=max(32,int(np.ceil(gh*3)))
            roi=crop_box((x1+x2)/2,(y1+y2)/2,rw,rh,W,H);bg=nonoverlap_background(gt,W,H,rw,rh)
            out={'image':ip.name,'gt_index':gi,'class_id':int(it[0]),'source_group':groups[ip.name]['source_group'],'gt_width':gw,'gt_height':gh,'tiny_1536':min(gw,gh)*.375<8}
            for name,b in [('defect',roi),('background',bg)]:
                o=im[b[1]:b[3],b[0]:b[2]];r=rec[b[1]:b[3],b[0]:b[2]];psnr,ssim=quality(o,r);out[f'{name}_psnr']=psnr;out[f'{name}_ssim']=ssim
                for k,v in responses(cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)).items():out[f'{name}_original_{k}']=v
                for k,v in responses(cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)).items():out[f'{name}_restored_{k}']=v
            for metric in ['gradient_mean','gradient_peak','horizontal_edge_mean','vertical_edge_mean','laplacian_abs_mean','intensity_std']:
                od=out[f'defect_original_{metric}'];ob=out[f'background_original_{metric}'];rd=out[f'defect_restored_{metric}'];rb=out[f'background_restored_{metric}']
                out[f'original_contrast_{metric}']=abs(od-ob)/(abs(ob)+1e-9);out[f'restored_contrast_{metric}']=abs(rd-rb)/(abs(rb)+1e-9);out[f'delta_contrast_{metric}']=out[f'original_contrast_{metric}']-out[f'restored_contrast_{metric}']
            droi=im[roi[1]:roi[3],roi[0]:roi[2]];dr=rec[roi[1]:roi[3],roi[0]:roi[2]];broi=im[bg[1]:bg[3],bg[0]:bg[2]];br=rec[bg[1]:bg[3],bg[0]:bg[2]]
            out['defect_edge_peak_shift_px']=edge_peak_shift(droi,dr);out['background_edge_peak_shift_px']=edge_peak_shift(broi,br)
            gx1=max(0,int(x1)-2);gy1=max(0,int(y1)-2);gx2=min(W,int(np.ceil(x2))+2);gy2=min(H,int(np.ceil(y2))+2);patch=im[gy1:gy2,gx1:gx2].astype(np.float32);patch_r=rec[gy1:gy2,gx1:gx2].astype(np.float32);out['gt_boundary_neighborhood_mae']=float(np.abs(patch-patch_r).mean())
            rows.append(out);examples.append((out['delta_contrast_gradient_mean'],ip.name,gi,im[roi[1]:roi[3],roi[0]:roi[2]],rec[roi[1]:roi[3],roi[0]:roi[2]]))
    df=pd.DataFrame(rows);df.to_csv(a.output/'information_loss_per_gt.csv',index=False)
    summary={}
    for key,g in [('all',df),('tiny',df[df.tiny_1536]),('flash',df[df.class_id==0]),('black',df[df.class_id==1])]:summary[key]={c:float(g[c].mean()) for c in ['defect_psnr','background_psnr','defect_ssim','background_ssim','delta_contrast_gradient_mean','delta_contrast_laplacian_abs_mean']}
    (a.output/'information_loss_summary.json').write_text(json.dumps({'protocol':{'downsample':'OpenCV INTER_AREA 4096x1024 -> 1536x384','upsample':'OpenCV INTER_LINEAR 1536x384 -> 4096x1024','roi':'GT-centered 3x box, minimum 32x32','background':'same-size deterministic non-overlapping offset'},'rows':len(df),'summary':summary},indent=2)+'\n')
    examples=sorted(examples,key=lambda x:x[0],reverse=True)[:12];canvas=np.full((len(examples)*100,400,3),255,np.uint8)
    for i,(delta,name,gi,o,r) in enumerate(examples):
        o=cv2.resize(o,(160,80));r=cv2.resize(r,(160,80));canvas[i*100:i*100+80,0:160]=o;canvas[i*100:i*100+80,170:330]=r;cv2.putText(canvas,f'{name} gt{gi} dC={delta:.3f}',(0,i*100+96),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
    cv2.imwrite(str(a.output/'roi_original_vs_restored.jpg'),canvas)
if __name__=='__main__':main()
