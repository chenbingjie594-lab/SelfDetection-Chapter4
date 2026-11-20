import csv,json,torch
from pathlib import Path
from chapter4_detection.models.rglcd import RGLCD
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage19'
def test_runtime_switch_exact():
 torch.manual_seed(4);a=RGLCD(2);b=RGLCD(2);b.load_state_dict(a.state_dict());a.gamma.data.fill_(.1);b.gamma.data.fill_(.1);b.diagnostics_enabled=True;x=torch.randn(1,2,8,9,requires_grad=True);z=x.detach().clone().requires_grad_(True);ya=a(x);yb=b(z);assert torch.equal(ya,yb);ya.sum().backward();yb.sum().backward();assert torch.equal(x.grad,z.grad) and torch.equal(a.phi.weight.grad,b.phi.weight.grad)
def test_feature_scope_and_groups():
 rows=list(csv.DictReader((OUT/'stage19_feature_instances.csv').open()));assert {int(r['imgsz']) for r in rows}=={640,1536};assert {r['state'] for r in rows}=={'INIT','TRAINED_E0'};assert {'G1_ORDINARY','G2_HIGH_CONTRAST','G3_HIGH_CLASS'}.issubset({r['kind'] for r in rows});assert all(0<=int(r['p3_y'])<int(r['p3_h']) for r in rows)
def test_decision_prevents_fabricated_cv():
 text=(OUT/'STAGE19_DECISION.md').read_text();assert 'FAIL_MECHANISM_RECHECK' in text;assert not (OUT/'RGLCD_STAGE19_FORMAL_FREEZE.json').exists();assert not (OUT/'RGLCD_GROUP_CV_RESULTS.md').exists()
def test_hard_review_pending():
 r=list(csv.DictReader((OUT/'HARD_BACKGROUND_REVIEW.csv').open()));assert len(r)==12;assert {x['human_status'] for x in r}=={'PENDING_REVIEW'};assert len(list((OUT/'hard_background_review').glob('*.jpg')))==12
