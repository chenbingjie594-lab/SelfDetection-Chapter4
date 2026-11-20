#!/usr/bin/env python3
"""Freeze selected Stage16 machine records and produce a deterministic hash manifest."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path

FILES = [
    "DATASET_LINEAGE.csv", "unknown_35_provenance.csv", "dataset_version_inventory.csv",
    "duplicate_and_source_relations.csv", "independent_test_candidates.csv",
    "lineage_summary.json", "independent_test_summary.json", "EXPERT_REVIEW_TEMPLATE.csv",
    "annotation_review_audit.csv", "annotation_package_metadata.json",
]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage16',type=Path,required=True);ap.add_argument('--protocol',type=Path,required=True)
    a=ap.parse_args();a.protocol.mkdir(parents=True,exist_ok=True)
    for name in FILES: shutil.copy2(a.stage16/name,a.protocol/name)
    hashes={}
    for root in (a.stage16,a.protocol):
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name!='FILE_HASHES.json': hashes[str(p)]=sha(p)
    payload={'algorithm':'SHA256','files':hashes}
    (a.stage16/'FILE_HASHES.json').write_text(json.dumps(payload,indent=2)+"\n")
    (a.protocol/'FILE_HASHES.json').write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({'copied_records':len(FILES),'hashed_files':len(hashes)},indent=2))
if __name__=='__main__':main()
