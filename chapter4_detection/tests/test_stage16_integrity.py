import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "stage16"


def rows(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def test_expert_package_is_pending_and_complete():
    review = rows("EXPERT_REVIEW_TEMPLATE.csv")
    assert len(review) == 25
    assert {r["review_status"] for r in review} == {"PENDING_EXPERT_REVIEW"}
    assert all(not r["reviewer_id"] and not r["corrected_bbox"] for r in review)
    assert len(list((OUT / "annotation_review").glob("*.jpg"))) == 25


def test_lineage_is_complete_and_split_groups_isolated():
    assert len(rows("DATASET_LINEAGE.csv")) == 184
    summary = json.loads((OUT / "lineage_summary.json").read_text())
    assert summary["current_split_counts"] == {"val": 46, "train": 138}
    assert summary["current_source_group_intersection"] == []
    assert summary["current_canonical_id_intersection"] == []
    assert summary["xml_trace_unknown"] == 35
    assert summary["historical_reconstruction_recovered"] == 25
    assert summary["still_unknown"] == 10


def test_no_test_candidate_is_silently_admitted():
    candidates = rows("independent_test_candidates.csv")
    assert candidates
    assert all(r["eligible"] == "NO" for r in candidates)
    summary = json.loads((OUT / "independent_test_summary.json").read_text())
    assert summary["verdict"] == "NO_VERIFIED_INDEPENDENT_TEST_SET"
    assert summary["eligible_images"] == 0


def test_original_dataset_manifest_is_only_an_input():
    meta = json.loads((OUT / "annotation_package_metadata.json").read_text())
    assert meta["inputs"]
    assert meta["expert_conclusions_completed"] == 0
