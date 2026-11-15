import json
from pathlib import Path

import pytest

from src.annotation.human_review import HumanReview, export_review_form, validation_report

QUEUE = Path("data/annotations/human_annotation_queue.jsonl")


def review(pair, who, decision="none"):
    return HumanReview(
        pair_id=pair,
        annotator_id=who,
        timestamp="2026-11-11T12:00:00+00:00",
        decision=decision,
        direction="not_applicable",
        independent_review=True,
        source="human_submission",
    )


def test_pending_and_blinded_export(tmp_path):
    assert validation_report(QUEUE, [])["status"] == "pending"
    export_review_form(QUEUE, tmp_path / "form.csv")
    assert "PROVISIONAL" not in (tmp_path / "form.csv").read_text()
    assert "annotator_id" in (tmp_path / "form.csv").read_text()


def test_agreement_and_duplicates():
    ids = [json.loads(line)["pair_id"] for line in QUEUE.read_text().splitlines()][:2]
    records = [
        review(ids[0], "a"),
        review(ids[1], "a", "association"),
        review(ids[0], "b"),
        review(ids[1], "b", "association"),
    ]
    report = validation_report(QUEUE, records)
    assert report["agreements"][0]["kappa"] == 1
    assert report["double_reviewed_items"] == 2
    assert validation_report(QUEUE, records[:1] + records[2:3])["agreements"][0]["kappa"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        validation_report(QUEUE, records + records[:1])
    with pytest.raises(ValueError):
        review("x", "", "causal")
    with pytest.raises(ValueError, match="unknown"):
        validation_report(QUEUE, [review("unknown", "a")])
