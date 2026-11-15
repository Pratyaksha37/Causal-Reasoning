"""Command-line input and review import integration checks."""

import csv
import json
import subprocess
import sys

import pytest

from src.annotation.human_review import (
    Adjudication,
    HumanReview,
    adjudicate,
    export_review_form,
    read_reviews,
)

QUEUE = "data/annotations/human_annotation_queue.jsonl"


def test_csv_review_and_adjudication(tmp_path):
    target = tmp_path / "form.csv"
    export_review_form(QUEUE, target)
    assert read_reviews(target) == []
    with target.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0].update(
        annotator_id="reviewer-a",
        timestamp="2026-11-11T10:00:00+00:00",
        decision="none",
        direction="not_applicable",
        independent_review="true",
        source="human_submission",
    )
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first = read_reviews(target)[0]
    second = first.model_copy(update={"annotator_id": "reviewer-b"})
    decision = Adjudication(
        pair_id=first.pair_id,
        adjudicator_id="senior",
        timestamp="2026-11-11T11:00:00+00:00",
        decision="none",
        direction="not_applicable",
        rationale="Synthetic unit test only",
        source="human_adjudication",
    )
    report = adjudicate(QUEUE, [first, second], [decision])
    assert report["adjudicated_items"] == 1
    assert len(report["gold_records"]) == 1
    assert report["gold_records"][0]["review_status"] == "HUMAN_ADJUDICATED"
    assert "polarity" not in report["gold_records"][0]
    assert not report["human_gold_established"]
    with pytest.raises(ValueError, match="two independent"):
        adjudicate(QUEUE, [first], [decision])
    with pytest.raises(ValueError):
        HumanReview(**{**first.model_dump(), "independent_review": False})


def test_cli_missing_data_and_pending_report():
    proc = subprocess.run([sys.executable, "-m", "src", "run"], capture_output=True, text=True)
    assert proc.returncode == 2
    assert "Supply --data" in proc.stderr
    proc = subprocess.run(
        [sys.executable, "-m", "src", "review-report"], capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["status"] == "pending"
