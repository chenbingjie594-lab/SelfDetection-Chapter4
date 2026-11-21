#!/usr/bin/env python3
"""Audit an already cross-fitted Stage23 quality-target file."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--targets',type=Path,required=True);p.add_argument('--outer-holdout',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.targets.read_text());r=d['records'];hold={Path(x).name for x in a.outer_holdout.read_text().splitlines()};overlap=sorted({x['image'] for x in r}&hold);assert not overlap,overlap;assert len({x['candidate_id'] for x in r})==len(r);assert all(len(x['geo'])==13 and len(x['full'])==205 for x in r);out={'records':len(r),'unique_candidates':len({x['candidate_id'] for x in r}),'outer_holdout_overlap':0,'inner_folds':sorted({x['inner_fold'] for x in r}),'quality_label_counts':{'useful':sum(x['positive'] and x['quality_label']==1 for x in r),'not_useful':sum(x['positive'] and x['quality_label']==0 for x in r),'background':sum(not x['positive'] for x in r)}};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
