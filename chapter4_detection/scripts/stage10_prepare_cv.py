#!/usr/bin/env python3
"""Materialize immutable train-only grouped-CV image lists and data YAMLs."""
import argparse,json,hashlib
from pathlib import Path
import yaml
EXPECTED='48054c137c8e093957c57eef6d761750cfac6f7b437e089cebe42053fb4b62ab'
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--folds',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 m=json.loads(a.manifest.read_text());f=json.loads(a.folds.read_text());assert m['protocol_sha256']==f['source_protocol_sha256']==EXPECTED
 rec={r['image_name']:r for r in m['records'] if r['split']=='train'};a.output_dir.mkdir(parents=True,exist_ok=True);seen=[]
 for q in f['folds']:
  hold=set(q['images']);train=sorted(set(rec)-hold);val=sorted(hold);assert not set(train)&set(val);seen+=val
  d=a.output_dir/f"fold{q['fold']}";d.mkdir(exist_ok=True)
  (d/'train.txt').write_text(''.join(rec[x]['image']+'\n' for x in train));(d/'holdout.txt').write_text(''.join(rec[x]['image']+'\n' for x in val))
  data={'path':str(d.resolve()),'train':str((d/'train.txt').resolve()),'val':str((d/'holdout.txt').resolve()),'names':{0:'Flash point',1:'Big black spots'}}
  (d/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
 assert sorted(seen)==sorted(rec) and len(seen)==len(set(seen))==138
 hashes={str(x.relative_to(a.output_dir)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(a.output_dir.rglob('*')) if x.is_file()}
 (a.output_dir/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n');print(json.dumps({'status':'PASS','folds':3,'images_once':len(seen)},indent=2))
if __name__=='__main__':main()
