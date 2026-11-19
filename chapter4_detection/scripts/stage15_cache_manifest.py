#!/usr/bin/env python3
"""Hash large local Stage15 candidate caches without committing them."""
import argparse,hashlib,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
 for f in sorted(a.root.rglob('*.npz')):
  h=hashlib.sha256();
  with f.open('rb') as x:
   while b:=x.read(1<<20):h.update(b)
  records.append({'path':str(f.relative_to(a.root)),'bytes':f.stat().st_size,'sha256':h.hexdigest()})
 a.output.write_text(json.dumps({'root':str(a.root.resolve()),'files':len(records),'total_bytes':sum(x['bytes'] for x in records),'records':records},indent=2)+'\n')
if __name__=='__main__':main()
