#!/usr/bin/env python3
"""Targeted triage of Stage16 LABELS_UNVERIFIED candidates; no global rescan."""
import argparse,csv,os
from collections import defaultdict
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidates',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 src=[r for r in csv.DictReader(a.candidates.open()) if r['status']=='LABELS_UNVERIFIED'];groups=defaultdict(list)
 for r in src:groups[str(Path(r['candidate_path']).parent)].append(r)
 out=[]
 for col,rs in sorted(groups.items()):
  p=Path(col); defect=p.name in {'01_Flash_point','02_Big_black_spots'}; masks=[]
  for r in rs:
   q=Path(r['candidate_path']); s=str(q).replace('/test/','/ground_truth/');c=[Path(s).with_name(q.stem+'_mask.png'),Path(s).with_suffix('.png')];masks.extend(str(x) for x in c if x.exists())
  split70='Plastic_Bomo_split70_seed42' in col
  label=('paired segmentation masks found for %d/%d images; not verified as human HBB'%(len(set(masks)),len(rs))) if defect else 'normal-image group; zero-object status is not source-verified'
  out.append({'collection':col,'image_count':len(rs),'candidate_label_source':label,'original_image_evidence':'MVTec-formatted derivative; original filename mapping not found in this group','acquisition_batch':'UNKNOWN','overlap_status':'no exact current-set match recorded by Stage16, but cross-format duplicates exist within candidate families','provenance_status':'UNKNOWN','annotation_status':'segmentation masks available; HBB provenance unverified' if masks else 'no trusted detection annotation','historical_use_status':'used in historical split70 train/eval' if split70 else 'historical MVTec anomaly dataset; development role not proven absent','candidate_priority':'MEDIUM_FOR_PROVENANCE_REVIEW' if masks and not split70 else 'LOW','blocking_reason':'REQUIRES_PROVENANCE_EVIDENCE; '+('also historical development use' if split70 else 'human HBB and acquisition independence unverified'),'final_status':'REQUIRES_PROVENANCE_EVIDENCE'})
 with (a.output/'TEST_CANDIDATE_GROUPS.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(f'{len(src)} candidates grouped into {len(out)} collections; verified independent: 0')
if __name__=='__main__':main()
