#!/usr/bin/env python3
"""Create the traceable Stage-6 reports from frozen analysis artifacts."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def f(x): return f"{x:.4f}"
def main():
 p=argparse.ArgumentParser();p.add_argument("--stage6",type=Path,required=True);p.add_argument("--repo",type=Path,default=Path("."));a=p.parse_args();s=a.stage6
 boot=json.loads((s/"paired_group_bootstrap.json").read_text()); audit=json.loads((s/"independent_test_audit.json").read_text()); loc=json.loads((s/"localization_error_analysis.json").read_text())
 # Intervention table and machine-readable aggregate.
 ir=[]
 for seed in (42,3407,2026):
  for mode in ("full","gate_bypass","residual_bypass"):
   base=s/"interventions"/f"e3_seed{seed}_{mode}";m=json.loads(Path(str(base)+"_metrics.json").read_text());d=json.loads(Path(str(base)+"_diagnostics.json").read_text());c=json.loads(Path(str(base)+"_confidence.json").read_text())
   ir.append({"seed":seed,"mode":mode,"map50_95":m["map50_95"],"ap75":m["map75"],"flash_ap50_95":m["per_class"]["0"]["ap50_95"],"black_ap50_95":m["per_class"]["1"]["ap50_95"],"black_recall":m["per_class"]["1"]["recall"],"black_tp_025":c["fixed_thresholds"]["0.25"]["1"]["tp"],"black_fp_025":c["fixed_thresholds"]["0.25"]["1"]["fp"],"black_fn_025":c["fixed_thresholds"]["0.25"]["1"]["fn"],"low_contrast_black_detected":d["groups"]["class1/contrast/low"]["detected"],"background_fp_025":sum(v for k,v in d["false_prediction_categories"].items() if k.endswith("/background"))})
 (s/"gate_intervention.json").write_text(json.dumps(ir,indent=2)+"\n")
 with (s/"gate_intervention.csv").open("w",newline="") as q:w=csv.DictWriter(q,fieldnames=ir[0],lineterminator="\n");w.writeheader();w.writerows(ir)
 # Independent audit.
 lines=["# Independent Test Audit","","## Decision","",f"**{audit['overall_status']} — no candidate qualifies as a formal independent test set.**","","The 46-image validation set has already guided model design and is not an independent test set. Historical directories named `test` or `eval` were audited conservatively; a directory name is not provenance.","","| Candidate | Images | Status | Exact overlaps | Known source-name overlaps | Near candidates (dHash≤8) |","|---|---:|---|---:|---:|---:|"]
 for z in audit["candidates"]:lines.append(f"| `{z['directory']}` | {z['image_count']} | {z['status']} | {len(z['exact_file_overlap'])} | {len(z.get('known_source_name_overlap',[]))} | {len(z.get('near_duplicate_candidates_hamming_le_8',[]))} |")
 lines += ["","The perceptual-hash screen found near-duplicate candidates in several legacy collections, but missing source provenance prevents deciding whether every pair is the same physical source. They remain `UNKNOWN`, not independent. Candidates with proven overlap would be `FAIL_OVERLAP`; all candidates here lack complete source-group provenance and proof of non-use during historical development. No formal test evaluation was run, and the frozen train/val split was not changed."]
 (s/"INDEPENDENT_TEST_AUDIT.md").write_text("\n".join(lines)+"\n")
 # Bootstrap report.
 lines=["# Paired Source-group Bootstrap Report","","Protocol: 5,000 paired resamples of 44 source groups (46 images), seed 20261115. AP is recomputed from duplicated per-image predictions; duplicates retain independent statistical copies. No replicate lacked either class.","","| Difference (seed mean) | Metric | Mean | 95% percentile CI | P(>0) |","|---|---|---:|---:|---:|"]
 for pair,z in boot["intervals"].items():
  for metric,q in z["seed_mean"].items():lines.append(f"| {pair} | {metric} | {f(q['mean'])} | [{f(q['ci95'][0])}, {f(q['ci95'][1])}] | {q['p_gt_0']:.3f} |")
 lines += ["","E3−E0 mAP50-95 is +0.0082, but its interval crosses zero. E3−E0 AP75 is +0.0349 with a wide interval crossing zero. A1−E3 AP75 is +0.0145 (the table expresses E3−A1 = −0.0145), also with a wide interval crossing zero. Thus neither localization claim is supported as dataset-sampling robust.","","These intervals condition on frozen checkpoints and the current validation population. They neither replace an independent test nor fully model retraining randomness."]
 (s/"PAIRED_GROUP_BOOTSTRAP_REPORT.md").write_text("\n".join(lines)+"\n")
 # Intervention report.
 lines=["# NBR-DE Gate Intervention","","No retraining was performed. Each E3 best checkpoint was evaluated under Full, Gate Bypass (`F+γD`), and Residual Bypass (`F`) using identical preprocessing/NMS.","","| Seed | Mode | mAP50-95 | AP75 | Black AP | Black Recall | Black TP/FP/FN @.25 | Low-contrast Black | Background FP @.25 |","|---:|---|---:|---:|---:|---:|---|---:|---:|"]
 for r in ir:lines.append(f"| {r['seed']} | {r['mode']} | {f(r['map50_95'])} | {f(r['ap75'])} | {f(r['black_ap50_95'])} | {f(r['black_recall'])} | {r['black_tp_025']}/{r['black_fp_025']}/{r['black_fn_025']} | {r['low_contrast_black_detected']}/13 | {r['background_fp_025']} |")
 lines += ["","Gate Bypass changes mAP50-95 by less than 0.001 for every seed. Residual Bypass is likewise nearly unchanged. Therefore the learned residual injection has little direct inference-time effect at the final checkpoints. Combined with the much poorer separately trained No-Gate ablation, the evidence is most consistent with the gate/module affecting training optimization or representation learning while the final explicit residual magnitude is small. The intervention creates distribution shift and is mechanism evidence, not standalone causal proof."]
 (s/"NBR_DE_GATE_INTERVENTION.md").write_text("\n".join(lines)+"\n")
 # Localization report and contributor count on common IoU50 matches.
 by={(z["model"],z["seed"],z["image"],z["gt_index"]):z for z in loc["instances"]}; comps=[]
 for seed in (42,3407,2026):
  dif=[]
  for k,z in by.items():
   if k[0]=="E3" and k[1]==seed and ("E0",seed,k[2],k[3]) in by:dif.append(z["iou"]-by[("E0",seed,k[2],k[3])]["iou"])
  comps.append({"seed":seed,"common":len(dif),"e3_better":sum(x>0 for x in dif),"e3_worse":sum(x<0 for x in dif),"mean_iou_delta":float(np.mean(dif))})
 (s/"localization_contributors.json").write_text(json.dumps(comps,indent=2)+"\n")
 lines=["# Localization and Error Comparison","","The analysis uses class-aware one-to-one matching. IoU rates use all 58 GT instances; center/width/height errors use IoU≥0.50 matches only. Fixed-confidence FP categories separate background, class confusion, insufficient localization, and duplicates.","","| Model | Seed | IoU≥.50 | IoU≥.75 | IoU≥.90 | Center px | Width rel. error | Height rel. error |","|---|---:|---:|---:|---:|---:|---:|---:|"]
 for r in loc["summary"]:lines.append(f"| {r['model']} | {r['seed']} | {f(r['iou50_rate'])} | {f(r['iou75_rate'])} | {f(r['iou90_rate'])} | {f(r['center_error_mean'])} | {f(r['width_rel_error_mean'])} | {f(r['height_rel_error_mean'])} |")
 lines += ["","At IoU≥0.75, E3 exceeds the corresponding E0 rate in all three seeds, but the bootstrap AP75 interval remains wide and crosses zero. Improvements are distributed across multiple matched targets rather than one target, yet the 58-instance sample is too small to exclude dominance by a small subset. A1's higher mean AP75 is not statistically secure. The saved per-instance CSV and five side-by-side figures provide the auditable target-level evidence."]
 (s/"LOCALIZATION_ERROR_COMPARISON.md").write_text("\n".join(lines)+"\n")
 # Recommendation and final summary.
 rec="# Model Selection Recommendation\n\n**Provisional architecture choice: A1 (difference + gate), with E3 retained as the full research comparison—not as a proven superior final model.**\n\nE3 has the largest validation mean among the mature variants, but E3−E0 (+0.0082) and E3−A1 (+0.0061) bootstrap intervals cross zero. The similarity-reference branch has no robust independent gain, whereas removing the gate during training is consistently harmful. A1 is therefore the more parsimonious statement of the supported mechanism. Its AP75 advantage over E3 is also uncertain, so it must not be described as accuracy-superior. E4 remains exploratory only.\n\nCurrent evidence supports a thesis claim that background-difference gating changes training behavior and can improve high-IoU localization on this development split. It does not yet support a strong generalization claim. The highest-priority missing evidence is a provenance-isolated, never-used real test cohort (or externally collected data), followed by a preregistered replication with more source groups.\n"
 (s/"MODEL_SELECTION_RECOMMENDATION.md").write_text(rec)
 final="# Stage 6 Final Summary\n\n1. E3−E0 mAP50-95 is +0.0082 under paired group bootstrap, 95% CI [−0.0208, 0.0355]; uncertainty is large.\n2. E3 AP75 is consistently higher by seed, but its paired CI crosses zero; target-level gains involve multiple instances but remain vulnerable to the small 58-target sample.\n3. Gate/residual interventions barely change final inference metrics, so direct residual injection is weak; the training ablation indicates the main effect is likely optimization/representation learning.\n4. A1 is the preferable lightweight, parsimonious core, but is not proven more accurate than E3 or E0.\n5. No `PASS_UNSEEN` independent real test set was found; formal test evaluation was skipped.\n6. Existing results support a cautious mechanism contribution, not a definitive generalized performance claim.\n7. The critical missing evidence is a new provenance-isolated real test cohort and preregistered external replication.\n\nNo model was retrained, no historical metric was modified, and no validation/test threshold was tuned.\n"
 (s/"STAGE6_FINAL_SUMMARY.md").write_text(final)
 commands="""# Stage-6 reproducibility commands\nPYTHONPATH=. /mnt/sda/cbj/miniconda3/envs/yolo11/bin/python chapter4_detection/scripts/audit_independent_test.py --root /mnt/sda/cbj/dataset/Plastic_Bomo --manifest /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/split_manifest.json --output chapter4_detection/results/stage6/independent_test_audit.json\nPYTHONPATH=. /mnt/sda/cbj/miniconda3/envs/yolo11/bin/python chapter4_detection/scripts/paired_group_bootstrap.py --manifest /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/split_manifest.json --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val --output-dir chapter4_detection/results/stage6 --repeats 5000 --seed 20261115\nbash chapter4_detection/scripts/run_stage6_interventions.sh\nPYTHONPATH=. /mnt/sda/cbj/miniconda3/envs/yolo11/bin/python chapter4_detection/scripts/localization_error_analysis.py --results chapter4_detection/results --labels /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/labels/val --images /mnt/sda/cbj/dataset/Plastic_Bomo/chapter4_real_only_v1/images/val --output-dir chapter4_detection/results/stage6\n"""
 (s/"RUN_COMMANDS.md").write_text(commands)
 files=sorted(q for q in s.rglob("*") if q.is_file() and q.name!="FILE_HASHES.sha256")
 (s/"FILE_HASHES.sha256").write_text("".join(f"{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(a.repo)}\n" for q in files))
 print(json.dumps({"reports":6,"intervention_rows":len(ir),"hashed_files":len(files)}))
if __name__=="__main__":main()
