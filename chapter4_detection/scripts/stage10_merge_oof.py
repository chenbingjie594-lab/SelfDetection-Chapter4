#!/usr/bin/env python3
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,nargs=3,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--model',required=True);a=p.parse_args();images={}
 for q in a.inputs:
  d=json.loads(q.read_text());over=set(images)&set(d['images']);
  if over:raise RuntimeError(f'duplicate OOF images: {sorted(over)}')
  images.update(d['images'])
 if len(images)!=138:raise RuntimeError(f'expected 138 OOF images, got {len(images)}')
 out={'schema_version':1,'model':a.model,'mode':'grouped_train_only_OOF','protocol':'each image predicted once by fold model trained without its source group; last.pt epoch150','images_count':len(images),'images':dict(sorted(images.items()))}
 a.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
