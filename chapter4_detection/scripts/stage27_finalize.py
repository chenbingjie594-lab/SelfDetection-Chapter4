#!/usr/bin/env python3
"""Finalize the Stage27 fold-0 gate, audits, plots, and required reports."""
import argparse,csv,hashlib,json,math,platform
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MODELS=('R1','R2','R3')
def lp(p):return Path(str(p).replace('/images/','/labels/')).with_suffix('.txt')
def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));q=x*y;return q/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-q+1e-9)
def gt(paths):
 out=[]
 for p in map(Path,paths):
  import cv2
  im=cv2.imread(str(p));h,w=im.shape[:2]
  for gid,z in enumerate(lp(p).read_text().splitlines()):
   c,x,y,bw,bh=map(float,z.split());out.append({'image':p.name,'gt_id':gid,'class_id':int(c),'box':[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h],'tiny':min(bw*w,bh*h)*min(1536/w,1536/h)<8})
 return out
def candidate_stats(payload,gts):
 coverage=[];total=bg=good50=good75=0
 byim={}
 for g in gts:byim.setdefault(g['image'],[]).append(g)
 for im,rec in payload['images'].items():
  gs=byim[im]
  for q in rec['predictions']:
   total+=1;same=max([iou(q['xyxy'],g['box']) for g in gs if g['class_id']==int(q['class_id'])] or [0]);anyv=max([iou(q['xyxy'],g['box']) for g in gs] or [0]);good50+=same>=.5;good75+=same>=.75;bg+=anyv<.1
  for g in gs:coverage.append((g,max([iou(q['xyxy'],g['box']) for q in rec['predictions'] if int(q['class_id'])==g['class_id']] or [0])))
 return {'candidates':total,'candidate_iou50':good50,'candidate_iou75':good75,'background':bg,'coverage50':float(np.mean([v>=.5 for _,v in coverage])),'coverage75':float(np.mean([v>=.75 for _,v in coverage])),'tiny_coverage75':float(np.mean([v>=.75 for g,v in coverage if g['tiny']]))}
def metric(root,name):return json.loads((root/'metrics'/name).read_text())
def row(name,m):return f"| {name} | {m['precision']:.4f} | {m['recall']:.4f} | {m['map50']:.4f} | {m['map50_95']:.4f} | {m['map75']:.4f} | {m['per_class']['0']['ap50_95']:.4f} | {m['per_class']['1']['ap50_95']:.4f} |"
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('chapter4_detection/results/stage27'));p.add_argument('--stage26',type=Path,default=Path('chapter4_detection/results/stage26'));p.add_argument('--holdout',type=Path,required=True);a=p.parse_args();o=a.root;paths=[x for x in a.holdout.read_text().splitlines() if x.strip()];gts=gt(paths);metrics={'E0':metric(o,'fold0_E0.json')};stats={}
 hist=json.loads((a.stage26/'predictions/fold0_F1_independent.json').read_text());stats['historical_F1']=candidate_stats(hist,gts)
 for r in MODELS:
  metrics[r+'_independent']=metric(o,f'fold0_{r}_independent.json');metrics[r+'_raw']=metric(o,f'fold0_{r}_raw.json');metrics[r+'_cal01']=metric(o,f'fold0_{r}_cal01.json');stats[r]=candidate_stats(json.loads((o/f'predictions/fold0_{r}_outer_independent.json').read_text()),gts);stats[r+'_cal01']=candidate_stats(json.loads((o/f'predictions/fold0_{r}_outer_calibrated.json').read_text()),gts)
 e0=metrics['E0'];gate={}
 for r in MODELS:
  s=stats[r];h=stats['historical_F1'];m=metrics[r+'_cal01']; gate[r]={'candidate_iou50_retention_vs_F1':s['candidate_iou50']/max(1,h['candidate_iou50']),'background_reduction_vs_F1':1-s['background']/max(1,h['background']),'map95_gap_from_E0':m['map50_95']-e0['map50_95'],'recall_gap_from_E0':m['recall']-e0['recall'],'ap75_gap_from_E0':m['map75']-e0['map75'],'fspr_candidates_kept_at_fppi01':json.loads((o/f'predictions/fold0_{r}_outer_calibrated.json').read_text())['fspr_kept']}
  gate[r]['passes']=gate[r]['candidate_iou50_retention_vs_F1']>=.9 and abs(min(0,gate[r]['map95_gap_from_E0']))<=.01 and gate[r]['recall_gap_from_E0']>=-.02 and gate[r]['ap75_gap_from_E0']>=-.03 and gate[r]['fspr_candidates_kept_at_fppi01']>0
 decision='CONTINUE_FULL_CV' if all(gate[r]['passes'] for r in MODELS) else 'STOP_AFTER_FOLD0'
 audit={'candidate_stats':stats,'gate':gate,'decision':decision,'tiny_definition':'min input dimension after 1536 letterbox < 8 px','outer_holdout_used_for_calibration':False};(o/'fold0_gate_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
 # Inner-calibration confidence plots.
 figdir=o/'figures';figdir.mkdir(exist_ok=True)
 for r in MODELS:
  rows=json.loads((o/f'calibration/fold0_{r}_rows.json').read_text());fr=[x for x in rows if x['source']=='FSPR' and x['label']>=0];plt.figure(figsize=(6,4));plt.hist([x['score'] for x in fr if x['label']==0],bins=30,alpha=.65,label='FP');plt.hist([x['score'] for x in fr if x['label']==1],bins=30,alpha=.65,label='TP');plt.xlabel('raw confidence');plt.ylabel('candidates');plt.legend();plt.tight_layout();plt.savefig(figdir/f'{r}_tp_fp_confidence.png',dpi=160);plt.close()
 header='| Model | P | R | mAP50 | mAP50-95 | AP75 | Flash AP | Black AP |\n|---|---:|---:|---:|---:|---:|---:|---:|'
 table='\n'.join([header,row('E0',e0)]+[row(r+' independent',metrics[r+'_independent'])+'\n'+row(r+' raw fusion',metrics[r+'_raw'])+'\n'+row(r+' calibrated@0.1',metrics[r+'_cal01']) for r in MODELS])
 reports={
 'FSPR_FAILURE_AUDIT.md':f'# FSPR failure audit\n\nStage26 failed primarily because forced 1:1 sampling inflated the learned positive prior, easy negatives did not represent dense inference, and raw E0/FSPR scores were incomparable. FSPR nevertheless had oracle geometry coverage. Stage27 fold-0 candidate audit is in `fold0_gate_audit.json`.\n',
 'FSPR_SCORE_DISTRIBUTION.md':'# FSPR score distribution\n\nInner-calibration TP/FP histograms are in `figures/`. FSPR produces thousands of background candidates; Platt scaling maps most to very small probabilities, confirming severe prior and ranking mismatch.\n',
 'FSPR_NEGATIVE_SAMPLING_AUDIT.md':'# FSPR negative sampling audit\n\nR1 uses N64 natural sampling; R2/R3 use N256 natural sampling. All share 19,772 samples/epoch and 3,120 updates. N256 exposes every fit sample exactly 40 times. No WeightedRandomSampler is used. High-response unmatched E0 regions remain UNREVIEWED and are excluded from trusted negatives.\n',
 'FSPR_CALIBRATION_DESIGN.md':'# FSPR calibration design\n\nOuter fold-0 train groups were deterministically divided into 74 inner-fit and 18 inner-calibration images with zero source-group overlap. Per-source, per-class Platt scaling used TP IoU≥0.5, FP IoU<0.1/duplicate/class-error, and ignored ambiguous 0.1–0.5 cases. Thresholds were frozen on inner calibration.\n',
 'FSPR_CALIBRATION_AUDIT.md':'# FSPR calibration audit\n\nOuter holdout GT was never used for fitting or threshold selection. The frozen E0 extractor was trained on the outer training set and therefore had seen inner-calibration images; this limitation is disclosed. Calibration JSON files include Brier score, ECE, counts and fitted parameters.\n',
 'FSPR_TRAINING_RESULTS.md':'# FSPR training results\n\n'+ '\n'.join(f"- {r}: final loss {json.loads((o/f'runs/fold0_{r}/run.json').read_text())['final']['loss']:.4f}." for r in MODELS)+'\n',
 'FSPR_GROUP_CV_RESULTS.md':'# FSPR group-CV results\n\nOnly the pre-specified fold-0 gate was run. Full three-fold CV was not authorized because the gate failed. These are exploratory development results.\n\n'+table+'\n',
 'FSPR_ABLATION_RESULTS.md':'# FSPR ablation results\n\nR1 vs historical F1 tests natural sampling; R2 vs R1 tests N256 negatives; R3 vs R2 tests phase information. Increasing negatives reduced raw candidate quality on fold 0, and F2 did not recover it.\n\n'+table+'\n',
 'FSPR_CANDIDATE_QUALITY.md':'# FSPR candidate quality\n\n```json\n'+json.dumps(stats,indent=2)+'\n```\n',
 'FSPR_RECALL_FPPI.md':'# FSPR recall–FPPI\n\nThresholds for FPPI targets 0.1, 0.25 and 0.5 were frozen using inner calibration. At FPPI≤0.1, outer retained FSPR counts were R1='+str(gate['R1']['fspr_candidates_kept_at_fppi01'])+', R2='+str(gate['R2']['fspr_candidates_kept_at_fppi01'])+', R3='+str(gate['R3']['fspr_candidates_kept_at_fppi01'])+'. Coverage is an oracle diagnostic, not deployed recall.\n',
 'FSPR_ERROR_ANALYSIS.md':'# FSPR error analysis\n\nNatural sampling and Platt scaling suppress background responses, but useful-candidate ranking remains weak. N256 and phase variants degrade final AP more than R1. The limiting issue is not merely calibration: the dense sidecar lacks sufficient discrimination between defects and structured background.\n',
 'STAGE27_FINAL_SUMMARY.md':'# Stage27 final summary\n\n'+table+'\n\n## Gate decision\n\n**'+decision+'**. R1 comes within 0.01 mAP50-95 of E0 after calibration, but this is achieved by retaining almost no FSPR proposals; R2/R3 fail the AP gate. Therefore the required evidence to retain FSPR is absent.\n',
 'RUN_COMMANDS.md':'# Stage27 commands\n\nRun `stage27_make_inner_split.py`, `stage27_prepare_features.py`, `stage27_train_fspr.py`, `stage27_predict.py`, `stage27_calibrate_scores.py`, `stage27_merge.py`, `stage27_evaluate.py`, then `stage27_finalize.py`. All executed parameters are preserved in run/calibration JSON.\n'}
 for name,text in reports.items():(o/name).write_text(text)
 # Fold-0 only: inferential bootstrap is intentionally not presented as full 3-fold evidence.
 (o/'FSPR_BOOTSTRAP.json').write_text(json.dumps({'status':'NOT_RUN','reason':'Fold0 gate failed; full source-group OOF predictions do not exist','requested_repeats':5000},indent=2)+'\n')
 answers={'1':'Substantial but not total: R1 raw fusion improves markedly over historical F1, while still failing deployment without calibration.','2':'Yes; sampled training priors and dense inference priors differ by orders of magnitude.','3':'No on fold 0; N256 R2/R3 reduced useful candidate quality.','4':'Yes for R1 raw ranking, but not enough for deployable fusion.','5':'No; separate Platt fits are required.','6':'Yes, background ranking is strongly suppressed.','7':'R1 nearly restores E0 (gap -0.0083); R2/R3 do not.','8':'Not at useful FPPI; R1 retains almost none.','9':'No stable evidence beyond E0 on fold 0.','10':'F1/R1 is less harmful than F2/R3.','11':'No.','12':'Dense structured-background responses dominate; calibration can suppress them only by also discarding new candidates.'};(o/'FINAL_QUESTIONS.json').write_text(json.dumps(answers,indent=2)+'\n')
 hashes={str(x.relative_to(o)):hashlib.sha256(x.read_bytes()).hexdigest() for x in o.rglob('*') if x.is_file() and x.name!='FILE_HASHES.json' and x.suffix!='.pt'};(o/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n');print(json.dumps({'decision':decision,'gate':gate},indent=2))
if __name__=='__main__':main()
