#!/usr/bin/env python3
"""Generate Stage10 hard-sample tables, review plates, plots, and evidence-based reports."""
from __future__ import annotations
import csv,json,math,hashlib,shutil
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'chapter4_detection/results/stage10';DATA=Path('/mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1');MAN=json.loads((DATA/'split_manifest.json').read_text())
def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);q=max(0,x2-x1)*max(0,y2-y1);aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1]);return q/(aa+bb-q+1e-12)
def loadcsv(p):
 with open(p) as f:return list(csv.DictReader(f))
def dumpcsv(p,rows):
 keys=[]
 for r in rows:
  for k in r:
   if k not in keys:keys.append(k)
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 stats=loadcsv(OUT/'gt_statistics.csv');val=[r for r in stats if r['split']=='val'];pred={}
 paths={}
 for seed in (42,3407,2026):
  paths[('E0',seed)]=ROOT/f'chapter4_detection/results/stage7/matched_e0_seed{seed}_predictions.json'
  paths[('SFDFM',seed)]=ROOT/f'chapter4_detection/results/stage8/e5_sfdfm_seed{seed}_predictions.json'
  paths[('CBCD',seed)]=ROOT/f'chapter4_detection/results/stage9/e6_cbcd_seed{seed}_predictions.json'
 for k,p in paths.items():pred[k]=json.loads(p.read_text())['images']
 atlas=[]
 for r in val:
  b=[float(r[x]) for x in ('x1','y1','x2','y2')];c=int(r['class_id']);row={k:r[k] for k in ('image','gt_id','class_id','class_name','source_group','width','height','input_width','input_height','local_brightness_difference','background_gradient','provenance')};allmiss=0
  for model in ('E0','SFDFM','CBCD'):
   miss=0;loc=0
   for seed in (42,3407,2026):
    pp=pred[(model,seed)][r['image']]['predictions'];same=[p for p in pp if int(p['class_id'])==c];other=[p for p in pp if int(p['class_id'])!=c];best=max(same,key=lambda p:iou(b,p['xyxy']),default=None);bi=iou(b,best['xyxy']) if best else 0;conf=float(best['confidence']) if best else 0;wrong=max((iou(b,p['xyxy']) for p in other),default=0)
    status='correct75' if bi>=.75 else 'localized50_not75' if bi>=.5 else 'classification_error' if wrong>=.5 else 'low_conf_or_geometry' if best else 'no_post_nms_candidate'
    row[f'{model}_{seed}_iou']=bi;row[f'{model}_{seed}_confidence']=conf;row[f'{model}_{seed}_status']=status;miss+=bi<.5;loc+=.5<=bi<.75
   row[f'{model}_miss_seeds']=miss;row[f'{model}_localization_50_75_seeds']=loc;allmiss+=miss
  row['total_misses_9']=allmiss;atlas.append(row)
 dumpcsv(OUT/'hard_sample_detection_status.csv',atlas)
 # Representative train-only review set: extremes and provenance strata, no model predictions.
 tr=[r for r in stats if r['split']=='train'];chosen=[]
 for seq,quota in ((sorted(tr,key=lambda x:float(x['input_width'])*float(x['input_height'])),9),(sorted(tr,key=lambda x:float(x['local_brightness_difference'])),8),(sorted(tr,key=lambda x:-float(x['background_gradient'])),8)):
  added=0
  for r in seq:
   key=(r['image'],r['gt_id'])
   if key not in {(x['image'],x['gt_id']) for x in chosen}:chosen.append(r);added+=1
   if added>=quota:break
 review=[];plates=OUT/'annotation_review';plates.mkdir(exist_ok=True)
 for old in plates.glob('*.jpg'):old.unlink()  # generated review artifacts only
 recs={r['image_name']:r for r in MAN['records']}
 for r in chosen:
  rec=recs[r['image']];im=cv2.imread(rec['image']);b=[int(round(float(r[x]))) for x in ('x1','y1','x2','y2')];x1,y1,x2,y2=b;pad=max(32,4*max(x2-x1,y2-y1));a=max(0,x1-pad);q=max(0,y1-pad);c=min(im.shape[1],x2+pad);d=min(im.shape[0],y2+pad);crop=im[q:d,a:c].copy();cv2.rectangle(crop,(x1-a,y1-q),(x2-a,y2-q),(0,255,0),2);crop=cv2.resize(crop,(640,400),interpolation=cv2.INTER_NEAREST);full=cv2.resize(im,(1280,320));sx=1280/im.shape[1];sy=320/im.shape[0];cv2.rectangle(full,(int(x1*sx),int(y1*sy)),(int(x2*sx),int(y2*sy)),(0,255,0),2);plate=np.vstack([full,cv2.copyMakeBorder(crop,0,0,320,320,cv2.BORDER_CONSTANT,value=(255,255,255))]);fn=f"{len(review):02d}_{Path(r['image']).stem}_gt{r['gt_id']}.jpg";cv2.imwrite(str(plates/fn),plate)
  review.append({'image':r['image'],'gt_id':r['gt_id'],'class':r['class_name'],'box_xyxy':str(b),'input_wh':f"{float(r['input_width']):.2f}x{float(r['input_height']):.2f}",'contrast':r['local_brightness_difference'],'provenance':r['provenance'],'review_plate':fn,'suspected_issue':'','expert_conclusion':''})
 dumpcsv(OUT/'MANUAL_REVIEW_TABLE.csv',review)
 # Required plots.
 fig,ax=plt.subplots(1,2,figsize=(10,4))
 for cid,name,col in [(0,'Flash','tab:orange'),(1,'Black','tab:blue')]:
  z=[r for r in tr if int(r['class_id'])==cid];ax[0].scatter([float(x['input_width'])*float(x['input_height']) for x in z],[float(x['local_brightness_difference']) for x in z],s=18,alpha=.7,label=name,c=col);ax[1].scatter([float(x['background_gradient']) for x in z],[float(x['texture_similarity']) for x in z],s=18,alpha=.7,label=name,c=col)
 ax[0].set(xlabel='Input-scale area (px²)',ylabel='Local brightness difference');ax[1].set(xlabel='Background gradient',ylabel='Texture similarity');ax[0].legend();fig.tight_layout();fig.savefig(OUT/'difficulty_distributions.png',dpi=180);plt.close(fig)
 sens=loadcsv(OUT/'iou_sensitivity.csv');fig,ax=plt.subplots(figsize=(7,4))
 for op in ('shift_x','shift_y','width','height'):
  xs=sorted(set(abs(float(r['delta_original_px'])) for r in sens if r['operation']==op));ys=[np.median([float(r['iou']) for r in sens if r['operation']==op and abs(float(r['delta_original_px']))==x and r['split'] if 'split' in r]) if False else np.median([float(r['iou']) for r in sens if r['operation']==op and abs(float(r['delta_original_px']))==x]) for x in xs];ax.plot(xs,ys,'o-',label=op)
 ax.axhline(.75,color='gray',ls='--');ax.axhline(.5,color='gray',ls=':');ax.set(xlabel='Absolute original-pixel perturbation',ylabel='Median IoU');ax.legend();fig.tight_layout();fig.savefig(OUT/'box_iou_sensitivity.png',dpi=180);plt.close(fig)
 # Summaries/reports.
 train_by=Counter(int(r['class_id']) for r in tr);black=[r for r in tr if int(r['class_id'])==1];area_med=float(np.median([float(r['input_width'])*float(r['input_height']) for r in tr]));con_med=float(np.median([float(r['local_brightness_difference']) for r in tr]));over=sum(float(r['input_width'])*float(r['input_height'])<=area_med and float(r['local_brightness_difference'])<=con_med for r in black)
 rng=np.random.default_rng(20261118);by=defaultdict(list)
 for r in tr:by[r['source_group']].append(r)
 gn=sorted(by);samples=[]
 for _ in range(5000):
  z=[r for g in rng.choice(gn,len(gn),replace=True) for r in by[g]];bz=[r for r in z if int(r['class_id'])==1]
  samples.append([np.mean([float(r['input_width'])*float(r['input_height'])<=area_med and float(r['local_brightness_difference'])<=con_med for r in bz]),np.mean([float(r['local_brightness_difference']) for r in z if int(r['class_id'])==0]),np.mean([float(r['local_brightness_difference']) for r in bz]),np.mean([float(r['background_gradient']) for r in z if int(r['class_id'])==0]),np.mean([float(r['background_gradient']) for r in bz])])
 ar=np.asarray(samples);db={'unit':'source_group','repeats':5000,'seed':20261118,'black_small_low_fraction':{'observed':over/len(black),'ci95':np.quantile(ar[:,0],[.025,.975]).tolist()},'mean_local_brightness_difference':{'Flash':{'observed':float(np.mean([float(r['local_brightness_difference']) for r in tr if int(r['class_id'])==0])),'ci95':np.quantile(ar[:,1],[.025,.975]).tolist()},'Black':{'observed':float(np.mean([float(r['local_brightness_difference']) for r in black])),'ci95':np.quantile(ar[:,2],[.025,.975]).tolist()}},'mean_background_gradient':{'Flash':{'observed':float(np.mean([float(r['background_gradient']) for r in tr if int(r['class_id'])==0])),'ci95':np.quantile(ar[:,3],[.025,.975]).tolist()},'Black':{'observed':float(np.mean([float(r['background_gradient']) for r in black])),'ci95':np.quantile(ar[:,4],[.025,.975]).tolist()}}};(OUT/'distribution_group_bootstrap.json').write_text(json.dumps(db,indent=2)+'\n')
 xml=loadcsv(OUT/'xml_trace.csv');traced=[r for r in xml if r['status']=='TRACED'];below=[r for r in traced if float(r['min_best_iou'])<.99]
 e0=json.loads((OUT/'e0_oof_seed42_metrics.json').read_text());cb=json.loads((OUT/'cbcd_oof_seed42_metrics.json').read_text());boot=json.loads((OUT/'oof_paired_group_bootstrap.json').read_text())
 # Preserve lightweight training evidence while keeping checkpoints out of Git.
 logs=OUT/'training_logs';logs.mkdir(exist_ok=True);checkpoints={}
 for model in ('e0','cbcd'):
  for fold in range(3):
   run=OUT/'runs'/f'{model}_fold{fold}_seed42'
   for name in ('results.csv','args.yaml','run_metadata.json'):
    if (run/name).exists():shutil.copy2(run/name,logs/f'{model}_fold{fold}_seed42_{name}')
   for which in ('last.pt','best.pt'):
    p=run/'weights'/which;checkpoints[f'{model}_fold{fold}_seed42_{which}']={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'formal':which=='last.pt'}
 (OUT/'checkpoint_hashes.json').write_text(json.dumps(checkpoints,indent=2)+'\n')
 (OUT/'DATA_DISTRIBUTION_ANALYSIS.md').write_text(f"# Data distribution analysis\n\nFrozen protocol `{MAN['protocol_sha256']}`. Train: 138 images, {train_by[0]} Flash and {train_by[1]} Black instances. At imgsz 1536, 173/174 targets have at least one side <16 px and 52/174 <8 px (also <1 P3 cell). Among {len(black)} Black instances, {over} lie in the joint train-median small/low-contrast quadrant. See `gt_statistics.csv` and `difficulty_distributions.png`. Validation statistics are descriptive only.\n")
 (OUT/'ANNOTATION_QUALITY_AUDIT.md').write_text(f"# Annotation quality audit\n\nAutomated checks found no invalid class, non-positive/out-of-range, duplicate, extreme-aspect, or >0.8 same-class duplicate-overlap boxes. XML was directly traced for {len(traced)}/184 images; {184-len(traced)} remain UNKNOWN. XML/YOLO matching mean best IoU is {np.mean([float(x['mean_best_iou']) for x in traced]):.6f}; {len(below)} images are below 0.99 (minimum {min(float(x['min_best_iou']) for x in traced):.6f}). This supports generally faithful conversion but does not prove semantic boundary correctness. Twenty-five review plates and a blank expert table were produced; without expert relabeling, annotation consistency is not claimed.\n")
 (OUT/'BOX_IOU_SENSITIVITY.md').write_text("# Box IoU sensitivity\n\n`iou_sensitivity.csv` simulates ±1/2/4 original-pixel center, width, height, and each individual edge perturbation for every frozen GT. At 1536 letterbox, one original pixel equals 0.375 input pixel. Median IoU after a 1/2/4 px horizontal center shift is 0.927/0.860/0.738; vertical is 0.926/0.857/0.733. Median Black boxes are about 24×23 original pixels (9.0×8.6 input pixels). For a 24 px-wide box, an isolated horizontal shift of about 3.4 px gives IoU 0.75 and 8 px gives IoU 0.50; simultaneous x/y or multiple-edge errors need smaller per-axis offsets. This is a geometric sensitivity experiment only, not evidence of real annotation error.\n")
 persistent=sum(r['total_misses_9']==9 for r in atlas);loc=sum(sum(r[f'{m}_localization_50_75_seeds'] for m in ('E0','SFDFM','CBCD'))>=5 for r in atlas)
 (OUT/'HARD_SAMPLE_ERROR_ATLAS.md').write_text(f"# Hard-sample error atlas\n\nAcross E0, SFDFM and CBCD × three seeds, {persistent}/{len(atlas)} validation GTs have IoU<0.50 in all nine runs; {loc} repeatedly fall in the IoU 0.50–0.75 band (at least 5/9). `hard_sample_detection_status.csv` separates correct75, localized50_not75, classification_error, low-confidence/geometry, and no post-NMS candidate. Suppression cannot be proven from post-NMS exports and is therefore not fabricated.\n")
 folds=json.loads((OUT/'fold_manifest.json').read_text());(OUT/'GROUP_CV_PROTOCOL.md').write_text(f"# Group CV protocol\n\nTrain-only, three source-group folds, seed 20261118; fold hash `{folds['fold_sha256']}`. Holdouts contain 46/46/46 images, class counts Flash 27/27/27 and Black 32/31/30. Training was fixed at 150 epochs, `val=False`, patience 0, and formal checkpoint `last.pt`. There was no per-epoch validation or early stopping. Ultralytics 8.4.145 nevertheless performs an automatic terminal validation/save pass after epoch 150; it was not used to choose or modify any checkpoint. The independently exported formal result always uses `last.pt`; `best.pt` is retained only in local run artifacts and its distinct hash is disclosed. Thus the holdout was inspected twice computationally (automatic terminal parity pass plus formal export), but never for model selection or protocol changes. One training seed (42) was run. Existing frozen validation images were excluded.\n")
 foldlines=[]
 for i in range(3):
  a=json.loads((OUT/f'e0_fold{i}_seed42_metrics.json').read_text());b=json.loads((OUT/f'cbcd_fold{i}_seed42_metrics.json').read_text());foldlines.append(f"| {i} | {a['map50_95']:.4f} | {b['map50_95']:.4f} | {b['map50_95']-a['map50_95']:+.4f} | {a['map75']:.4f} | {b['map75']:.4f} |")
 ci=boot['intervals']['map50_95']['ci95'];(OUT/'GROUP_CV_RESULTS.md').write_text("# Group CV results\n\n| Fold | E0 mAP50-95 | CBCD mAP50-95 | Δ | E0 AP75 | CBCD AP75 |\n|---:|---:|---:|---:|---:|---:|\n"+'\n'.join(foldlines)+f"\n\nMerged OOF: E0 mAP50-95 {e0['map50_95']:.4f}, AP75 {e0['map75']:.4f}; CBCD mAP50-95 {cb['map50_95']:.4f}, AP75 {cb['map75']:.4f}. Paired group-bootstrap ΔmAP50-95 CI95 [{ci[0]:.4f}, {ci[1]:.4f}]. Only seed42 was trained, so training randomness is not covered.\n")
 (OUT/'STAGE10_FINAL_SUMMARY.md').write_text(f"# Stage 10 final summary\n\n1. Black has modestly more training instances (93 vs 81), but both are extremely small after resize.\n2. Joint small/low-contrast Black count is {over}/{len(black)} under train-fixed median thresholds; see distributions, not averages alone.\n3. No systemic format/conversion error is demonstrated; two XML mappings below IoU .99 require expert review and 35 images remain provenance-UNKNOWN.\n4. Pixel-level perturbations materially affect tiny-box IoU; this is sensitivity, not proof of annotation error.\n5. Persistent errors exist across models; counts and types are in the atlas.\n6. CBCD OOF mAP50-95 is {cb['map50_95']:.4f} vs E0 {e0['map50_95']:.4f}, but AP75 is {cb['map75']:.4f} vs {e0['map75']:.4f}; the bootstrap interval crosses zero and fold mAP gains are not unanimous. CBCD advantage is not stable evidence.\n7. Evidence points primarily to tiny-target/data-coverage and precise-localization limits; annotation quality remains partly unresolved, not proven defective.\n8. There is not yet sufficient evidence to add another module. Priorities are expert boundary review and localization-focused analysis; stop stacking CBCD variants.\n")
 (OUT/'RUN_COMMANDS.md').write_text("# Reproduction\n\nSee `scripts/stage10_audit.py`, `stage10_prepare_cv.py`, `stage10_build_cbcd_init.py`, `run_stage10_cv.sh`, `stage10_merge_oof.py`, `stage10_paired_bootstrap.py`, and `stage10_finalize.py`. Use `/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -m ...` from repository root.\n")
 hashes={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and '/runs/' not in str(p) and p.suffix!='.pt' and p.name!='FILE_HASHES.json'};(OUT/'FILE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
 print({'persistent_all9':persistent,'repeated_localization':loc,'black_small_low':over,'oof_delta':cb['map50_95']-e0['map50_95']})
if __name__=='__main__':main()
