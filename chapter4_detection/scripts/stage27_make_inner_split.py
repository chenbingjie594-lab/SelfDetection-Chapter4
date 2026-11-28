#!/usr/bin/env python3
"""Create deterministic source-group isolated inner-fit/calibration lists."""
import argparse, hashlib, json, random
from collections import Counter
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--outer-train',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=42); p.add_argument('--fraction',type=float,default=.2); a=p.parse_args()
    paths={Path(x).name:x for x in a.outer_train.read_text().splitlines() if x.strip()}; records=json.loads(a.manifest.read_text())['records']; by_name={r['image_name']:r for r in records}
    missing=sorted(set(paths)-set(by_name));
    if missing: raise RuntimeError(f'missing provenance for {missing}')
    groups={by_name[n]['source_group'] for n in paths}; ordered=sorted(groups); random.Random(a.seed).shuffle(ordered); n=max(1,round(len(ordered)*a.fraction)); cal_groups=set(ordered[:n])
    cal=[paths[n] for n in sorted(paths) if by_name[n]['source_group'] in cal_groups]; fit=[paths[n] for n in sorted(paths) if by_name[n]['source_group'] not in cal_groups]
    a.output.mkdir(parents=True,exist_ok=True); (a.output/'inner_fit.txt').write_text('\n'.join(fit)+'\n'); (a.output/'inner_calibration.txt').write_text('\n'.join(cal)+'\n')
    payload={'seed':a.seed,'fraction':a.fraction,'outer_images':len(paths),'source_groups':len(groups),'fit_images':len(fit),'calibration_images':len(cal),'fit_groups':len(groups)-len(cal_groups),'calibration_groups':len(cal_groups),'calibration_source_groups':sorted(cal_groups),'overlap':sorted({by_name[Path(x).name]['source_group'] for x in fit}&cal_groups)}
    payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(); (a.output/'inner_split.json').write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
