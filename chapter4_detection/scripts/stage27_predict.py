#!/usr/bin/env python3
"""Generate raw FSPR candidates and optional calibrated E0+FSPR fusion."""
import argparse, json, math, sys
from pathlib import Path
from chapter4_detection.scripts import stage26_predict_fspr as raw_predict

def cal(score,p):
    s=min(max(score,1e-6),1-1e-6); z=p['a']*math.log(s/(1-s))+p['b']; return 1/(1+math.exp(-max(-50,min(50,z))))
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1])); q=ix*iy
    return q/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-q+1e-9)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',type=Path,required=True); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--images-list',type=Path,required=True); p.add_argument('--e0-oof',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--calibrator',type=Path); p.add_argument('--target-fppi',choices=('0.1','0.25','0.5'),default='0.25'); a=p.parse_args()
    old=sys.argv; sys.argv=['stage26_predict_fspr.py','--weights',str(a.weights),'--checkpoint',str(a.checkpoint),'--images-list',str(a.images_list),'--e0-oof',str(a.e0_oof),'--output',str(a.output)]
    try: raw_predict.main()
    finally: sys.argv=old
    if not a.calibrator: return
    cfg=json.loads(a.calibrator.read_text())['sources']; e0=json.loads(a.e0_oof.read_text()); fspr=json.loads(a.output.with_name(a.output.stem+'_independent.json').read_text()); fused={}; kept_total=0
    threshold=cfg['FSPR']['thresholds'][a.target_fppi]['threshold']
    for name,item in fspr['images'].items():
        base=[]
        for q in e0['images'][name]['predictions']:
            z=dict(q); z['raw_confidence']=z['confidence']; z['confidence']=cal(z['confidence'],cfg['E0']['classes'][str(z['class_id'])]); z['source']='E0'; base.append(z)
        added=[]
        for q in item['predictions']:
            z=dict(q); z['raw_confidence']=z['confidence']; z['confidence']=cal(z['confidence'],cfg['FSPR']['classes'][str(z['class_id'])])
            if z['confidence']<threshold: continue
            if any(int(b['class_id'])==z['class_id'] and iou(b['xyxy'],z['xyxy'])>=.7 for b in base): continue
            added.append(z)
        kept_total+=len(added); fused[name]={'width':item['width'],'height':item['height'],'predictions':base+added}
    payload={'schema_version':1,'model':'E0+FSPR-calibrated','images_count':len(fused),'images':fused,'calibration':str(a.calibrator.resolve()),'target_fppi':a.target_fppi,'fspr_threshold':threshold,'fspr_kept':kept_total,'outer_gt_used':False}
    a.output.with_name(a.output.stem+'_calibrated.json').write_text(json.dumps(payload)+'\n'); print(json.dumps({k:payload[k] for k in ('target_fppi','fspr_threshold','fspr_kept','outer_gt_used')},indent=2))
if __name__=='__main__': main()
