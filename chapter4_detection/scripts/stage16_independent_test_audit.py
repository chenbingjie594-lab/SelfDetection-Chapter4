#!/usr/bin/env python3
"""Apply conservative independent-test admission rules to historical test-labelled data."""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
from PIL import Image

EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write_csv(path,rows,fields):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--roots',type=Path,nargs='+',required=True);ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 recs=json.loads((a.dataset/'split_manifest.json').read_text())['records']; bysha={r['image_sha256']:r for r in recs}; byname={r['image_name']:r for r in recs}
 candidates=[]
 for root in a.roots:
  for p in root.rglob('*'):
   if not p.is_file() or p.suffix.lower() not in EXT: continue
   parts=[x.lower() for x in p.parts]
   if 'test' not in parts: continue
   digest=sha(p); match=bysha.get(digest); name_match=byname.get(p.name)
   label_candidates=[]
   ps=str(p)
   if '/images/' in ps: label_candidates.append(Path(ps.replace('/images/','/labels/')).with_suffix('.txt'))
   label=next((x for x in label_candidates if x.exists()),None)
   reasons=[]
   if match:
    status='OVERLAP_WITH_TRAIN' if match['split']=='train' else 'OVERLAP_WITH_VAL'
    reasons.append(f"exact SHA256 match to current {match['split']} image {match['image_name']}")
   elif name_match:
    status='DERIVED_FROM_EXISTING_SOURCE';reasons.append(f"same filename/canonical candidate as current {name_match['split']} record; exact file differs")
   elif 'empty_' in p.name.lower():
    status='PROVENANCE_UNKNOWN';reasons.append('background/empty crop has no verified acquisition batch, source group, and detection-label provenance')
   elif not label:
    status='LABELS_UNVERIFIED';reasons.append('no paired YOLO label found')
   else:
    status='PROVENANCE_UNKNOWN';reasons.append('directory split name is not documentary proof of an unseen acquisition source')
   candidates.append({'candidate_path':str(p),'candidate_collection':str(root),'sha256':digest,'label_path':str(label or ''),
    'status':status,'matched_current_image':match['image_name'] if match else (name_match['image_name'] if name_match else ''),
    'matched_current_split':match['split'] if match else (name_match['split'] if name_match else ''),
    'canonical_image_id':match['canonical_image_id'] if match else (name_match['canonical_image_id'] if name_match else 'UNKNOWN'),
    'source_group':match['source_group'] if match else (name_match['source_group'] if name_match else 'UNKNOWN'),
    'reason':' ; '.join(reasons),'eligible':'NO'})
 fields=['candidate_path','candidate_collection','sha256','label_path','status','matched_current_image','matched_current_split','canonical_image_id','source_group','reason','eligible']
 write_csv(a.output/'independent_test_candidates.csv',candidates,fields)
 status=Counter(r['status'] for r in candidates)
 summary={'verdict':'NO_VERIFIED_INDEPENDENT_TEST_SET','candidate_images':len(candidates),'status_counts':dict(status),'eligible_images':0,
  'admission_requirements':['new real production/acquisition batch','never used in model or generator development','original unmodified images and complete human detection labels','acquisition time and batch provenance','traceable canonical image IDs and source groups','hash/source-group non-overlap evidence against all development data','one-time evaluation only after method and hyperparameters are frozen'],
  'note':'A folder named test is not accepted as evidence of independence.'}
 (a.output/'independent_test_summary.json').write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
