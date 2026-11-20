#!/usr/bin/env python3
"""Validate expert decisions and emit proposals only; never edits source labels."""
import argparse,csv,json
from pathlib import Path
ALLOWED={'PENDING_EXPERT_REVIEW','CONFIRMED_CORRECT','BBOX_TOO_TIGHT','BBOX_TOO_LOOSE','CLASS_ERROR','FALSE_ANNOTATION','MISSING_OBJECT','AMBIGUOUS','NEED_MORE_CONTEXT'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 recs=json.loads((a.dataset/'split_manifest.json').read_text())['records'];by={r['image_name']:r for r in recs};rows=list(csv.DictReader(a.review.open()));seen=set();errors=[];proposals=[]
 for index,r in enumerate(rows,1):
  rid=r['review_id'];
  if rid != f'S17-{index:03d}':errors.append(f'{rid}: review_id/order mismatch, expected S17-{index:03d}')
  if rid in seen:errors.append(f'{rid}: duplicate review_id')
  seen.add(rid)
  if r['image'] not in by:errors.append(f'{rid}: image mismatch/not in manifest');continue
  if r['review_status'] not in ALLOWED:errors.append(f'{rid}: invalid status')
  if r['corrected_class'] and r['corrected_class'] not in {'0','1','Flash point','Big black spots'}:errors.append(f'{rid}: invalid class')
  if r['corrected_bbox']:
   try:
    b=[float(x) for x in r['corrected_bbox'].replace('[','').replace(']','').split(',')];w,h=by[r['image']]['width'],by[r['image']]['height']
    if len(b)!=4 or not (0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h):raise ValueError
   except ValueError:errors.append(f'{rid}: corrected_bbox invalid/out of bounds')
  if r['review_status']!='PENDING_EXPERT_REVIEW':proposals.append({k:r[k] for k in r})
 status='PENDING_HUMAN_REVIEW' if not proposals and not errors else ('VALIDATION_FAILED' if errors else 'PROPOSALS_ONLY_NO_LABELS_CHANGED')
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'expert_review_import_report.json').write_text(json.dumps({'status':status,'rows':len(rows),'proposals':proposals,'errors':errors,'labels_modified':False},indent=2)+"\n");print(status)
 raise SystemExit(1 if errors else 0)
if __name__=='__main__':main()
