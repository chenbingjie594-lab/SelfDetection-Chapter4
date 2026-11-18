import json
from pathlib import Path
from chapter4_detection.scripts.stage10_audit import box_iou

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'chapter4_detection/results/stage10'

def test_box_iou_pixel_shift():
    b=(0,0,10,10)
    assert abs(box_iou(b,b)-1.0)<1e-12
    assert abs(box_iou(b,(1,0,11,10))-9/11)<1e-12

def test_group_cv_manifest_is_disjoint_and_complete():
    d=json.loads((OUT/'fold_manifest.json').read_text())
    images=[set(f['images']) for f in d['folds']]
    groups=[set(f['source_groups']) for f in d['folds']]
    assert [len(x) for x in images]==[46,46,46]
    assert len(set.union(*images))==138 and not any(images[i]&images[j] for i in range(3) for j in range(i))
    assert len(set.union(*groups))==137 and not any(groups[i]&groups[j] for i in range(3) for j in range(i))
    assert [f['statistics']['c0'] for f in d['folds']]==[27,27,27]

def test_oof_has_each_train_image_once():
    for model in ('e0','cbcd'):
        d=json.loads((OUT/f'{model}_oof_seed42_predictions.json').read_text())
        assert d['images_count']==len(d['images'])==138
