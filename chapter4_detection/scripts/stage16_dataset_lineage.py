#!/usr/bin/env python3
"""Inventory Plastic_Bomo dataset versions and trace the frozen 184-image protocol."""
from __future__ import annotations
import argparse, csv, hashlib, json, os
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def pixel_hash(path):
    im=Image.open(path).convert('RGB')
    h=hashlib.sha256(); h.update(f'{im.width}x{im.height}:RGB'.encode()); h.update(im.tobytes())
    return h.hexdigest()

def dhash(path, n=16):
    im=Image.open(path).convert('L').resize((n+1,n))
    px=list(im.getdata()); value=0
    for y in range(n):
        for x in range(n): value=(value<<1)|(px[y*(n+1)+x]>px[y*(n+1)+x+1])
    return f'{value:0{n*n//4}x}'

def dataset_version(path, root):
    rel=path.relative_to(root)
    parts=rel.parts
    if 'images' in parts:
        i=parts.index('images'); base=parts[:i]
        # variant directory plus split is represented separately.
        return str(Path(root.name,*base))
    return str(Path(root.name,parts[0] if len(parts)>1 else '.'))

def split_name(path):
    parts=path.parts
    if 'images' in parts:
        i=parts.index('images')
        return parts[i+1] if i+2 < len(parts) else 'unspecified'
    return 'unspecified'

def write_csv(path, rows, fields=None):
    fields=fields or (list(rows[0]) if rows else [])
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,required=True)
    ap.add_argument('--roots',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.dataset/'split_manifest.json').read_text())
    records=manifest['records']; current_sha={r['image_sha256']:r for r in records}
    current_pixel={r['decoded_rgb_sha256']:r for r in records}
    current_names=defaultdict(list)
    for r in records: current_names[r['image_name']].append(r)

    files=[]; version_stats=defaultdict(lambda:{'images':0,'splits':Counter(),'labels':0,'xml':0,'classes':Counter()})
    sha_groups=defaultdict(list); current_matches=defaultdict(list)
    for root in a.roots:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix.lower() not in EXT: continue
            ver=dataset_version(p,root); sp=split_name(p); digest=sha(p)
            row={'image_path':str(p),'dataset_version':ver,'split':sp,'sha256':digest,
                 'current_exact_match':current_sha.get(digest,{}).get('image_name','')}
            files.append(row); sha_groups[digest].append(str(p)); version_stats[ver]['images']+=1; version_stats[ver]['splits'][sp]+=1
            if digest in current_sha: current_matches[current_sha[digest]['image_name']].append(str(p))
        for p in root.rglob('*.xml'):
            # Attribute XML to its nearest top-level version.
            rel=p.relative_to(root); ver=str(Path(root.name,rel.parts[0] if len(rel.parts)>1 else '.')); version_stats[ver]['xml']+=1
        for p in root.rglob('*.txt'):
            if 'labels' not in p.parts: continue
            rel=p.relative_to(root); parts=rel.parts
            ver=str(Path(root.name,*parts[:parts.index('labels')])) if 'labels' in parts else str(Path(root.name,parts[0]))
            version_stats[ver]['labels']+=1
            try:
                for line in p.read_text().splitlines():
                    if line.strip(): version_stats[ver]['classes'][line.split()[0]]+=1
            except (UnicodeDecodeError,ValueError): pass
    write_csv(a.output/'all_image_file_hashes.csv',files)
    versions=[]
    for ver,s in sorted(version_stats.items()):
        versions.append({'dataset_version':ver,'image_count':s['images'],'label_file_count':s['labels'],'xml_count':s['xml'],
                         'splits':json.dumps(dict(s['splits']),sort_keys=True),'class_instances':json.dumps(dict(s['classes']),sort_keys=True),
                         'provenance_note':'Directory inventory only; version name does not establish provenance or independence.'})
    write_csv(a.output/'dataset_version_inventory.csv',versions)

    traces={r['image']:r for r in csv.DictReader((Path('chapter4_detection/results/stage10/xml_trace.csv')).open())}
    lineage=[]; unknown_detail=[]
    for r in records:
        prov=r.get('provenance',{}); method=prov.get('method','UNKNOWN')
        if method=='CONFIRMED_EXACT_HISTORICAL_RECONSTRUCTION': status='CONFIRMED_HISTORICAL_RECONSTRUCTION'
        elif method=='CONFIRMED_ORIGINAL_XML': status='CONFIRMED_ORIGINAL_XML'
        else: status='UNKNOWN'
        matches=current_matches.get(r['image_name'],[])
        evidence=prov.get('evidence','')
        relation='EXACT_DUPLICATES_ACROSS_VERSIONS' if len(matches)>1 else 'CURRENT_FROZEN_RECORD'
        row={'image_path':r['image'],'dataset_version':'chapter4_real_only_v1','canonical_image_id':r['canonical_image_id'],
             'source_group':r['source_group'],'source_xml':traces[r['image_name']]['xml'],'sha256':r['image_sha256'],
             'pixel_hash':r['decoded_rgb_sha256'],'perceptual_hash':r.get('dhash',''),'provenance_status':status,
             'duplicate_group':'sha256:'+r['image_sha256'][:16],'relation_type':relation,
             'evidence':evidence+' | exact paths: '+' ; '.join(matches),'confidence':'HIGH' if status!='UNKNOWN' else 'LOW'}
        lineage.append(row)
        if traces[r['image_name']]['status']=='UNKNOWN':
            unknown_detail.append({'image':r['image_name'],'split':r['split'],'xml_trace_status':'UNKNOWN',
                 'recovered_provenance_status':status,'evidence':evidence,'original_source':prov.get('original_source',''),
                 'original_source_sha256':prov.get('original_source_sha256',''),'current_sha256':r['image_sha256'],
                 'exact_historical_copies':len(matches),'conclusion':('Source recovered from exact historical reconstruction record; original XML remains unavailable.' if status=='CONFIRMED_HISTORICAL_RECONSTRUCTION' else 'UNKNOWN')})
    write_csv(a.output/'DATASET_LINEAGE.csv',lineage)
    write_csv(a.output/'unknown_35_provenance.csv',unknown_detail)

    dup=[]
    for digest,paths in sha_groups.items():
        if len(paths)>1 and digest in current_sha:
            rec=current_sha[digest]
            for p in paths:
                dup.append({'match_type':'EXACT_FILE_SHA256','group_id':digest[:16],'current_image':rec['image_name'],
                            'current_split':rec['split'],'canonical_image_id':rec['canonical_image_id'],'source_group':rec['source_group'],
                            'matched_path':p,'evidence':digest,'confidence':'EXACT'})
    write_csv(a.output/'duplicate_and_source_relations.csv',dup)
    train_groups={r['source_group'] for r in records if r['split']=='train'}; val_groups={r['source_group'] for r in records if r['split']=='val'}
    train_ids={r['canonical_image_id'] for r in records if r['split']=='train'}; val_ids={r['canonical_image_id'] for r in records if r['split']=='val'}
    summary={'scanned_roots':[str(x) for x in a.roots],'image_files_scanned':len(files),'dataset_versions':len(versions),
             'current_records':len(records),'current_split_counts':dict(Counter(r['split'] for r in records)),
             'current_source_group_intersection':sorted(train_groups&val_groups),'current_canonical_id_intersection':sorted(train_ids&val_ids),
             'xml_trace_unknown':len(unknown_detail),'historical_reconstruction_recovered':sum(x['recovered_provenance_status']=='CONFIRMED_HISTORICAL_RECONSTRUCTION' for x in unknown_detail),
             'still_unknown':sum(x['recovered_provenance_status']=='UNKNOWN' for x in unknown_detail),
             'exact_duplicate_relation_rows':len(dup),
             'limitations':['pHash alone is not used to assert lineage.','Crop/near-duplicate relations without documentary or local-match confirmation remain UNKNOWN.']}
    (a.output/'lineage_summary.json').write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
