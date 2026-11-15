#!/usr/bin/env python3
"""Export E3 predictions under inference-only NBR-DE interventions."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
from ultralytics import YOLO
from chapter4_detection.models.nbr_de import NBRDE, register_nbr_de
from chapter4_detection.scripts.export_validator_predictions import PredictionCaptureValidator

def main():
 p=argparse.ArgumentParser(); p.add_argument("--weights",type=Path,required=True); p.add_argument("--data",type=Path,required=True); p.add_argument("--images",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--mode",choices=("full","gate_bypass","residual_bypass"),required=True); p.add_argument("--device",default="0"); p.add_argument("--seed",type=int,required=True); a=p.parse_args()
 register_nbr_de(); model=YOLO(str(a.weights)); modules=[m for m in model.model.modules() if isinstance(m,NBRDE)]
 if not modules: raise RuntimeError("Checkpoint contains no NBRDE")
 for m in modules: m.inference_intervention=a.mode
 PredictionCaptureValidator.captured={}
 metrics=model.val(validator=PredictionCaptureValidator,data=str(a.data),imgsz=1536,batch=1,device=a.device,conf=.001,iou=.7,max_det=300,plots=False,verbose=False)
 records={}
 for ip in sorted(a.images.iterdir()):
  if ip.suffix.lower() not in {".jpg",".jpeg",".png"}: continue
  im=cv2.imread(str(ip)); h,w=im.shape[:2]; records[ip.name]={"width":w,"height":h,"predictions":PredictionCaptureValidator.captured.get(ip.name,[])}
 payload={"schema_version":1,"model":f"E3_seed{a.seed}_{a.mode}","intervention":a.mode,"weights":str(a.weights.resolve()),"validator_metrics":{"map50":float(metrics.box.map50),"map50_95":float(metrics.box.map)},"images":records}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+"\n")
if __name__=="__main__": main()
