#!/usr/bin/env python3
"""Correct Stage22 candidate coverage without altering Stage22 artifacts."""
import argparse,json
from pathlib import Path
from chapter4_detection.scripts.stage23_common import load_gt,best_rows,summarize

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage10',type=Path,default=Path('chapter4_detection/results/stage10'));p.add_argument('--stage22',type=Path,default=Path('chapter4_detection/results/stage22'));p.add_argument('--gt-stats',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 gt=[r for r in load_gt(a.gt_stats) if r['split']=='train'];base=a.stage10/'e0_oof_seed42_predictions.json';models={}
 for name,disk in [('A','A'),('B','B'),('C','C'),('C-noDiff','C_noDiff')]:
  rows=best_rows(base,a.stage22/f'{disk}_oof_predictions.json',gt);models[name]={'summary':summarize(rows),'rows':rows}
 e0=best_rows(base,base,gt);out={'protocol':'oracle maximum IoU over all same-class post-NMS candidates, independently selected before and after refinement','E0':{'summary':summarize(e0),'rows':e0},**models}
 (a.output/'STAGE22_CORRECTED_CANDIDATE_METRICS.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Stage22 evaluation correction','','Stage22 fixed the original best-candidate index when measuring refined coverage. That metric remains useful as `fixed_candidate_iou_change`, but it is not GT coverage after refinement. This correction independently re-selects the maximum-IoU same-class post-NMS candidate for every GT. It is an offline oracle diagnostic and is never used at inference.','','| Model | Coverage@.75 | Mean best IoU | Tiny coverage | Gains | Losses | ID switches |','|---|---:|---:|---:|---:|---:|---:|']
 for n in ['E0','A','B','C','C-noDiff']:
  q=out[n]['summary'];lines.append(f"| {n} | {q['coverage75_after_count']}/{q['gt']} | {q['mean_best_iou_after']:.4f} | {q['tiny_coverage75_after_count']}/{q['tiny_gt']} | {q['gain75']} | {q['lose75']} | {q['best_id_switches']} |")
 lines+=['','The original Stage22 STOP decision is preserved. This correction does not retroactively alter its frozen joint decision rule or reports.']
 (a.output/'STAGE22_EVALUATION_CORRECTION.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
