#!/usr/bin/env python3
"""Prepare the final 27-item blind expert handoff without changing annotations."""
import argparse,csv,json,shutil
from pathlib import Path
from PIL import Image,ImageDraw
from stage16_annotation_package import parse_yolo,parse_xml,expanded,panel

EXTRA=['img_1013_pre_part_1.jpg','img_9509_pre_part_0.jpg']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--stage16',type=Path,required=True);ap.add_argument('--xml-trace',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 out=a.output/'expert_review';out.mkdir(parents=True,exist_ok=True)
 recs=json.loads((a.dataset/'split_manifest.json').read_text())['records']; by={r['image_name']:r for r in recs}; traces={r['image']:r for r in csv.DictReader(a.xml_trace.open())}
 old=list(csv.DictReader((a.stage16/'EXPERT_REVIEW_TEMPLATE.csv').open())); rows=[]
 for i,r in enumerate(old):
  src=next((p for p in (a.stage16/'annotation_review').glob(f'*_{Path(r["image"]).stem}_gt{r["gt_id"]}.jpg')),None); name=f'S17-{i+1:03d}_{src.name}'
  shutil.copy2(src,out/name);rows.append({'review_id':f'S17-{i+1:03d}','image':r['image'],'gt_id':r['gt_id'],'plate':name,'review_status':'PENDING_EXPERT_REVIEW','reviewer_id':'','review_date':'','object_valid':'','class_correct':'','boundary_quality':'','corrected_class':'','corrected_bbox':'','confidence':'','reason':'','notes':''})
 for name in EXTRA:
  r=by[name];im=Image.open(r['image']).convert('RGB'); boxes=parse_yolo(Path(r['label']),*im.size); cls,box=boxes[0]; xp=Path(traces[name]['xml']);_,xb=parse_xml(xp); xbox=xb[0][1];crop=expanded(box,*im.size)
  head=Image.new('RGB',(1400,100),'white');d=ImageDraw.Draw(head);d.text((10,10),f'{name} | GT 0 | Big black spots',fill='black');d.text((10,40),f'YOLO xyxy: {[round(v,2) for v in box]} | XML: {list(xbox)}',fill='black');d.text((10,70),'Blind review: no model prediction or metric shown.',fill='black')
  canvas=Image.new('RGB',(1400,530),'white');canvas.paste(head,(0,0));canvas.paste(panel(im,crop,box,(230,30,30),'View A - current YOLO annotation'),(0,100));canvas.paste(panel(im,crop,xbox,(30,80,230),'View B - historical XML annotation'),(700,100))
  i=len(rows);plate=f'S17-{i+1:03d}_{Path(name).stem}_gt0.jpg';canvas.save(out/plate,quality=94)
  rows.append({'review_id':f'S17-{i+1:03d}','image':name,'gt_id':'0','plate':plate,'review_status':'PENDING_EXPERT_REVIEW','reviewer_id':'','review_date':'','object_valid':'','class_correct':'','boundary_quality':'','corrected_class':'','corrected_bbox':'','confidence':'','reason':'','notes':''})
 with (a.output/'STAGE17_EXPERT_REVIEW.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(f'prepared {len(rows)} pending reviews')
if __name__=='__main__':main()
