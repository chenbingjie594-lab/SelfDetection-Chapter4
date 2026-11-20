#!/usr/bin/env python3
"""Write Stage17 SHA256 manifest after all audit-only outputs are complete."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();files={str(p):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file() and p.name!='FILE_HASHES.json'};(a.output/'FILE_HASHES.json').write_text(json.dumps({'algorithm':'SHA256','files':files},indent=2)+"\n");print(len(files))
if __name__=='__main__':main()
