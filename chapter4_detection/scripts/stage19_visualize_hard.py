#!/usr/bin/env python3
"""Create review plates for highest-response train-only hard-background candidates."""
import argparse,csv
from pathlib import Path
from PIL import Image,ImageDraw
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--instances',type=Path,required=True);ap.add_argument('--train-list',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--count',type=int,default=12);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 paths={Path(x).name:Path(x) for x in a.train_list.read_text().splitlines()};rows=[r for r in csv.DictReader(a.instances.open()) if r['state']=='TRAINED_E0' and r['imgsz']=='1536' and r['kind']=='G3_HIGH_CLASS'];rows=sorted(rows,key=lambda r:float(r['class_response']),reverse=True)[:a.count];out=[]
 for i,r in enumerate(rows,1):
  im=Image.open(paths[r['image']]).convert('RGB');x_in=(float(r['p3_x'])+.5)*8;y_in=(float(r['p3_y'])+.5)*8;ox=x_in/0.375;oy=(y_in-576)/0.375;side=320;box=(max(0,int(ox-side)),max(0,int(oy-side)),min(im.width,int(ox+side)),min(im.height,int(oy+side)));crop=im.crop(box).resize((900,500));d=ImageDraw.Draw(crop);cx=(ox-box[0])*900/(box[2]-box[0]);cy=(oy-box[1])*500/(box[3]-box[1]);d.ellipse((cx-12,cy-12,cx+12,cy+12),outline='red',width=4);d.rectangle((0,0,900,45),fill='white');d.text((8,8),f"{r['image']} HARD_BACKGROUND_CANDIDATE score={float(r['class_response']):.4f} D={float(r['d_abs']):.3f} r={float(r['r_mean']):.3f}",fill='black');plate=f'HB-{i:02d}_{Path(r["image"]).stem}.jpg';crop.save(a.output/plate,quality=94);out.append({'candidate_id':f'HB-{i:02d}','image':r['image'],'original_x':round(ox,2),'original_y':round(oy,2),'class_response':r['class_response'],'d_abs':r['d_abs'],'r_mean':r['r_mean'],'suppression':r['suppression'],'plate':plate,'human_status':'PENDING_REVIEW','is_true_background':'','notes':''})
 with (a.output.parent/'HARD_BACKGROUND_REVIEW.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(len(out))
if __name__=='__main__':main()
