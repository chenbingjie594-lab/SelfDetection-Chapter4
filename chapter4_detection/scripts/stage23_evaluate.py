#!/usr/bin/env python3
"""Evaluate Stage23 best-candidate localization with independent candidate re-selection."""
import argparse,json
from pathlib import Path
from chapter4_detection.scripts.stage23_common import load_gt,best_rows,summarize
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--prediction',type=Path,required=True);p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--images-list',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();allowed={Path(x).name for x in a.images_list.read_text().splitlines()};rows=best_rows(a.base,a.prediction,load_gt(a.gt_stats,allowed));out={'summary':summarize(rows),'rows':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))
if __name__=='__main__':main()
