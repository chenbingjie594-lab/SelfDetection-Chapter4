#!/usr/bin/env python3
"""Materialize auditable B/C hard-background candidates and review plates."""
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np,torch
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plates-per-fold',type=int,default=24);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);plate=a.output/'review_plates';plate.mkdir(exist_ok=True);rows=[]
 for f in range(3):rows.extend(torch.load(a.cache_dir/f'fold{f}.pt',map_location='cpu',weights_only=False)['hard_candidates'])
 fields=list(rows[0]);
 with (a.output/'hard_background_candidates.csv').open('w',newline='') as z:w=csv.DictWriter(z,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 for f in range(3):
  sel=sorted([r for r in rows if r['fold']==f],key=lambda r:r['e0_score'],reverse=True)[:a.plates_per_fold]
  for i,r in enumerate(sel):
   im=cv2.imread(r['image_path']);h,w=im.shape[:2];cx,cy=r['original_x'],r['original_y'];side=512;x1=max(0,int(cx-side/2));y1=max(0,int(cy-side/2));x2=min(w,x1+side);y2=min(h,y1+side);native=im[y1:y2,x1:x2];native=cv2.resize(native,(640,320))
   small=cv2.resize(im,(1536,384));canvas=np.full((1536,1536,3),114,np.uint8);canvas[576:960]=small;ix=(r['p3_x']+.5)*8;iy=(r['p3_y']+.5)*8;s=192;inp=canvas[max(0,int(iy-s)):min(1536,int(iy+s)),max(0,int(ix-s)):min(1536,int(ix+s))];inp=cv2.resize(inp,(640,320));out=np.concatenate((native,inp),axis=1);text=f"fold{f} {r['image']} cls={r['predicted_class']} score={r['e0_score']:.3f} {r['review_status']}";cv2.rectangle(out,(0,0),(1280,35),(255,255,255),-1);cv2.putText(out,text,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),1,cv2.LINE_AA);cv2.imwrite(str(plate/f'fold{f}_{i:02d}_{Path(r["image"]).stem}.jpg'),out,[cv2.IMWRITE_JPEG_QUALITY,92])
 counts={};
 for f in range(3):counts[str(f)]={'B':sum(r['fold']==f and r['review_status'].startswith('B_') for r in rows),'C':sum(r['fold']==f and r['review_status'].startswith('C_') for r in rows),'A':0}
 (a.output/'hard_background_summary.json').write_text(json.dumps({'counts':counts,'policy':'A=human-confirmed (none available); B=weak unconfirmed weight .25; C=excluded suspected defect/ambiguity','plates':len(list(plate.glob('*.jpg')))},indent=2)+'\n')
 (a.output/'HARD_BACKGROUND_MINING_AUDIT.md').write_text('# Hard-background mining audit\n\nCandidates are mined strictly from each outer-fold training list and the fold-specific E0 decoded pre-NMS output. Positions inside twice-expanded GT regions, padding and TAL positives are excluded, and every retained decoded box has maximum IoU below 0.10 to every annotated GT. Scores in `[0.05,0.25)` are **B: unconfirmed conservative candidates** and receive weak weight 0.25; scores `>=0.25` are **C: possible missed defect/ambiguous** and are excluded. The CSV records the restored original box, original/P3 location, E0 class/score, maximum GT IoU and nearest-GT center distance. No independent human-review record was available, so **A=0** and no candidate is represented as confirmed normal background. Review plates show the native-resolution crop (left) and the corresponding 1536-square YOLO-input crop (right).\n\n'+ '\n'.join(f"- Fold {f}: A=0, B={counts[str(f)]['B']}, C={counts[str(f)]['C']}." for f in range(3))+'\n')
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
