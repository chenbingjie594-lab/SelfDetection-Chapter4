#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();d=json.loads(a.input.read_text());a.output_dir.mkdir(parents=True,exist_ok=True)
 for kind in ('froc','pr'):
  fig,ax=plt.subplots(figsize=(7,5))
  for n,rows in d.items():
   if kind=='froc':x=[r['background_fppi'] for r in rows];y=[r['recall'] for r in rows];ax.set_xlabel('Background-style false positives per image')
   else:
    y=[r['recall'] for r in rows];x=[]
    for r in rows:
     fp=sum(r['counts'].values())-r['counts'].get('stage24_legacy_sameclass_iou_lt010',0);x.append(r['matched']/(r['matched']+fp) if r['matched']+fp else 1)
    ax.set_xlabel('Precision (unified post-NMS taxonomy)')
   ax.plot(x,y,marker='o',ms=3,label=n)
  ax.set_ylabel('Recall @ IoU 0.5');ax.grid(alpha=.3);ax.legend(ncol=2);fig.tight_layout();fig.savefig(a.output_dir/f'{kind.upper()}_CURVE.png',dpi=180);plt.close(fig)
if __name__=='__main__':main()
