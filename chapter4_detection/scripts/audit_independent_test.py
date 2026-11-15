#!/usr/bin/env python3
"""Conservative audit of historical directories that could be mistaken for a test set."""
from __future__ import annotations
import argparse, hashlib, json
from multiprocessing import Pool
from pathlib import Path
import cv2

EXT={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def fingerprint(p):
 im=cv2.imread(str(p),cv2.IMREAD_COLOR)
 if im is None:return None,None
 gray=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(32,8),interpolation=cv2.INTER_AREA)
 comp=gray[:,:-1]>gray[:,1:]
 dhash=sum(int(comp[j,i]) << (i*8+j) for i in range(31) for j in range(8))
 # Match the frozen protocol: despite the historical field name, the hash is
 # over OpenCV's decoded BGR bytes.
 return hashlib.sha256(im.tobytes()).hexdigest(),dhash
def main():
 a=argparse.ArgumentParser(); a.add_argument("--root",type=Path,required=True); a.add_argument("--manifest",type=Path,required=True); a.add_argument("--output",type=Path,required=True); x=a.parse_args()
 records=json.loads(x.manifest.read_text())["records"]; known_sha={r["image_sha256"]:r for r in records}; known_rgb={r["decoded_rgb_sha256"]:r for r in records}; known_stem={Path(r["image_name"]).stem:r for r in records}
 candidate_dirs=sorted({p for p in x.root.rglob("*") if p.is_dir() and p.name.lower() in {"test","eval"}})
 out=[]; hash_cache={}; rgb_cache={}; known_dhash=[(int(r["dhash"]),r) for r in records]
 for d in candidate_dirs:
  imgs=sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in EXT); exact=[]; decoded=[]; near=[]; source_name=[]
  for p in imgs:
   if str(p) not in hash_cache: hash_cache[str(p)]=sha(p)
   s=hash_cache[str(p)]
   if s in known_sha: exact.append({"candidate":str(p),"known":known_sha[s]["image_name"],"split":known_sha[s]["split"]})
   if p.stem in known_stem:
    z=known_stem[p.stem]; source_name.append({"candidate":str(p),"known":z["image_name"],"source_group":z["source_group"],"split":z["split"]})
  # Exact overlap is already a conclusive FAIL. Pixel decoding is reserved for
  # candidates that survive the cheaper byte-level check.
  if not exact and not source_name:
   pending=[p for p in imgs if str(p) not in rgb_cache]
   if pending:
    with Pool(processes=min(8,len(pending))) as pool:
     for p,value in zip(pending,pool.map(fingerprint,pending)): rgb_cache[str(p)]=value
   for p in imgs:
    rs,dh=rgb_cache[str(p)]
    if rs in known_rgb: decoded.append({"candidate":str(p),"known":known_rgb[rs]["image_name"],"split":known_rgb[rs]["split"]})
    if dh is not None:
     distance,z=min((((dh^kd).bit_count(),r) for kd,r in known_dhash),key=lambda item:item[0])
     if distance<=8: near.append({"candidate":str(p),"known":z["image_name"],"source_group":z["source_group"],"hamming_distance":distance})
  # PASS requires provenance, source-group isolation and proof of no development use;
  # none of the legacy directories carries those three records together.
  status="FAIL_OVERLAP" if exact or decoded or source_name else "UNKNOWN"
  out.append({"directory":str(d),"image_count":len(imgs),"exact_file_overlap":exact,"known_source_name_overlap":source_name,"decoded_pixel_overlap":decoded,"near_duplicate_candidates_hamming_le_8":near,"decoded_check":"skipped_after_conclusive_exact_or_source_name_overlap" if exact or source_name else "completed","has_xml":sum(1 for _ in d.rglob("*.xml")),"status":status,"reason":"overlap with frozen development set/source groups" if status=="FAIL_OVERLAP" else "no complete source-group provenance and no evidence of non-use during historical development"})
 payload={"overall_status":"UNKNOWN" if not any(z["status"]=="PASS_UNSEEN" for z in out) else "PASS_UNSEEN","decision":"No candidate qualifies for formal independent testing.","criteria":"PASS_UNSEEN requires real-image proof, compatible labels, SHA/pixel/source-group isolation, provenance, and evidence of no historical model-development use.","candidates":out}
 x.output.parent.mkdir(parents=True,exist_ok=True); x.output.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"overall":payload["overall_status"],"candidates":len(out),"fail":sum(z['status']=='FAIL_OVERLAP' for z in out),"unknown":sum(z['status']=='UNKNOWN' for z in out)},indent=2))
if __name__=="__main__":main()
