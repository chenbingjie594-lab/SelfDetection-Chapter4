import csv,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/stage17'
def read(name):
 with (OUT/name).open() as f:return list(csv.DictReader(f))
def test_review_handoff():
 r=read('STAGE17_EXPERT_REVIEW.csv');assert len(r)==27;assert [x['review_id'] for x in r]==[f'S17-{i:03d}' for i in range(1,28)];assert {x['review_status'] for x in r}=={'PENDING_EXPERT_REVIEW'};assert len(list((OUT/'expert_review').glob('*.jpg')))==27
def test_candidate_triage():
 g=read('TEST_CANDIDATE_GROUPS.csv');assert sum(int(x['image_count']) for x in g)==681;assert all(x['final_status']!='VERIFIED_INDEPENDENT' for x in g)
def test_import_is_non_mutating_pending():
 x=json.loads((OUT/'expert_review_import_report.json').read_text());assert x['status']=='PENDING_HUMAN_REVIEW';assert x['labels_modified'] is False
