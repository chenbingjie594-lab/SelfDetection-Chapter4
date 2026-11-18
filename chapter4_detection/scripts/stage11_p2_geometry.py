#!/usr/bin/env python3
"""OOF P2/P3 boundary-versus-background geometric feature diagnosis."""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
import cv2,numpy as np,torch
from ultralytics import YOLO
def gt(label,W,H):
 out=[]
 for i,line in enumerate(label.read_text().splitlines()):
  z=line.split();c=int(z[0]);x,y,w,h=map(float,z[1:]);out.append((i,c,np.array([(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H])))
 return out
def regions(field,b,stride):
 H,W=field.shape;x1,y1,x2,y2=b/stride;x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2);yy,xx=np.indices((H,W));inside=(xx+.5>=x1)&(xx+.5<=x2)&(yy+.5>=y1)&(yy+.5<=y2)
 dist=np.minimum.reduce((abs(xx+.5-x1),abs(xx+.5-x2),abs(yy+.5-y1),abs(yy+.5-y2)));near=(xx+.5>=x1-1)&(xx+.5<=x2+1)&(yy+.5>=y1-1)&(yy+.5<=y2+1);boundary=near&(dist<=1);outer=near&~inside
 def mean(m):return float(field[m].mean()) if m.any() else float('nan')
 return mean(inside),mean(boundary),mean(outer)
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage10',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='0');a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);device=torch.device(f'cuda:{a.device}' if a.device.isdigit() else a.device);rows=[];identity=[]
 folds=json.loads((a.stage10/'fold_manifest.json').read_text())['folds'];rng=random.Random(20261118)
 for fold in range(3):
  wrapper=YOLO(str(a.stage10/f'runs/e0_fold{fold}_seed42/weights/last.pt'));model=wrapper.model.to(device).eval();cap={}
  hs=[model.model[2].register_forward_hook(lambda m,i,o:cap.__setitem__('p2',o.detach())),model.model[16].register_forward_hook(lambda m,i,o:cap.__setitem__('p3',o.detach()))]
  # cv2/cv3 receive the identical P3 tensor by stock Detect semantics.
  hs += [model.model[-1].cv2[0][0].register_forward_pre_hook(lambda m,i:cap.__setitem__('reg_in',i[0].detach())),model.model[-1].cv3[0][0].register_forward_pre_hook(lambda m,i:cap.__setitem__('cls_in',i[0].detach()))]
  for name in folds[fold]['images']:
   ip=a.data_root/'images/train'/name;im=cv2.imread(str(ip));H,W=im.shape[:2];gain=.375;res=cv2.resize(im,(1536,384));inp=cv2.copyMakeBorder(res,16,16,16,16,cv2.BORDER_CONSTANT,value=(114,114,114));x=torch.from_numpy(inp[:,:,::-1].copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
   with torch.no_grad():model(x)
   identity.append(float((cap['reg_in']-cap['cls_in']).abs().max().cpu()))
   feats={'P2':cap['p2'][0].float(),'P3':cap['p3'][0].float()}
   boxes=gt(a.data_root/'labels/train'/f'{Path(name).stem}.txt',W,H)
   for gi,c,b0 in boxes:
    b=b0*gain+np.array([16,16,16,16]);bw=b[2]-b[0];bh=b[3]-b[1]
    # Deterministic matched-size normal ROI, non-overlapping any GT, closest mean brightness.
    target=im[int(b0[1]):int(np.ceil(b0[3])),int(b0[0]):int(np.ceil(b0[2]))];cands=[]
    for _ in range(200):
     x1=rng.randrange(0,max(1,int(W-(b0[2]-b0[0]))));y1=rng.randrange(0,max(1,int(H-(b0[3]-b0[1]))));q=np.array([x1,y1,x1+(b0[2]-b0[0]),y1+(b0[3]-b0[1])])
     if all((q[2]<=z[2][0] or q[0]>=z[2][2] or q[3]<=z[2][1] or q[1]>=z[2][3]) for z in boxes):cands.append(q)
    bg=min(cands,key=lambda q:abs(float(im[int(q[1]):int(np.ceil(q[3])),int(q[0]):int(np.ceil(q[2]))].mean())-float(target.mean()))) if cands else None
    row={'fold':fold,'image':name,'gt_id':gi,'class_id':c,'width_input':bw,'height_input':bh}
    for level,f in feats.items():
     # Channel-RMS activation and spatial gradient magnitude.
     act=torch.sqrt((f*f).mean(0)+1e-12);gx=torch.diff(f,dim=2,prepend=f[:,:,:1]);gy=torch.diff(f,dim=1,prepend=f[:,:1,:]);edge=torch.sqrt((gx*gx+gy*gy).mean(0)+1e-12)
     for kind,field in [('activation',act),('edge',edge)]:
      inn,bd,out=regions(field.cpu().numpy(),b,4 if level=='P2' else 8);row[f'{level}_{kind}_inside']=inn;row[f'{level}_{kind}_boundary']=bd;row[f'{level}_{kind}_outer']=out
      if bg is not None:
       bb=bg*gain+np.array([16,16,16,16]);_,bbd,_=regions(field.cpu().numpy(),bb,4 if level=='P2' else 8);row[f'{level}_{kind}_matched_bg_boundary']=bbd;row[f'{level}_{kind}_specificity']=bd/(bbd+1e-12)
     # Peak offset inside a 2-cell context around target.
     stride=4 if level=='P2' else 8;x1=max(0,int(b[0]/stride)-2);y1=max(0,int(b[1]/stride)-2);x2=min(edge.shape[1],int(np.ceil(b[2]/stride))+2);y2=min(edge.shape[0],int(np.ceil(b[3]/stride))+2);patch=edge[y1:y2,x1:x2];ind=int(patch.argmax());py,px=np.unravel_index(ind,patch.shape);peak=np.array([(x1+px+.5)*stride,(y1+py+.5)*stride]);center=np.array([(b[0]+b[2])/2,(b[1]+b[3])/2]);row[f'{level}_edge_peak_center_offset']=float(np.linalg.norm(peak-center))
    rows.append(row)
  for h in hs:h.remove()
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with (a.output_dir/'p2_p3_geometry.csv').open('w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
 summary={'instances':len(rows),'reg_cls_input_max_abs':max(identity),'levels':{}}
 for lv in ('P2','P3'):
  summary['levels'][lv]={'edge_specificity_median':float(np.nanmedian([r[f'{lv}_edge_specificity'] for r in rows])),'edge_specificity_gt1_fraction':float(np.nanmean([r[f'{lv}_edge_specificity']>1 for r in rows])),'edge_peak_center_offset_mean':float(np.nanmean([r[f'{lv}_edge_peak_center_offset'] for r in rows])),'boundary_over_outer':float(np.nanmean([r[f'{lv}_edge_boundary']/(r[f'{lv}_edge_outer']+1e-12) for r in rows]))}
 (a.output_dir/'p2_p3_geometry_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(6,4));ax.boxplot([[r[f'{lv}_edge_specificity'] for r in rows] for lv in ('P2','P3')],labels=['P2','P3'],showfliers=False);ax.axhline(1,color='gray',ls='--');ax.set(ylabel='Defect boundary / matched-background edge response');fig.tight_layout();fig.savefig(a.output_dir/'p2_p3_edge_specificity.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
