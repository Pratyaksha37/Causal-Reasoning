"""Independent review records, kept separate from Phase 1 pseudo-labels."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
from sklearn.metrics import cohen_kappa_score


class HumanReview(BaseModel):
    """A reviewer-submitted decision; independence is an attestation, not verified fact."""

    model_config = ConfigDict(extra="forbid")
    pair_id: str
    annotator_id: str
    timestamp: datetime
    decision: Literal["causal", "association", "none", "uncertain"]
    independent_review: Literal[True]
    source: Literal["human_submission"]
    direction: Literal["entity_1->entity_2", "entity_2->entity_1", "not_applicable"]
    notes: str = ""

    @model_validator(mode="after")
    def validate_record(self) -> "HumanReview":
        if not self.annotator_id.strip() or not self.pair_id.strip():
            raise ValueError("Reviewer and pair identifiers must be nonempty")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Review timestamp must include a timezone")
        if self.decision == "causal" and self.direction == "not_applicable":
            raise ValueError("A causal decision requires direction")
        if self.decision != "causal" and self.direction != "not_applicable":
            raise ValueError("Only causal decisions may specify direction")
        return self


def load_queue(path: str | Path) -> dict[str, dict]:
    """Load the existing candidate queue, rejecting ambiguous duplicate identifiers."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    queue = {r["pair_id"]: r for r in records}
    if len(queue) != len(records):
        raise ValueError("Duplicate pair identifiers in queue")
    return queue


def export_review_form(queue_path: str | Path, destination: str | Path) -> None:
    """Export a blinded blank form, deliberately excluding all pseudo-labels."""
    rows = []
    for pair_id, item in load_queue(queue_path).items():
        rows.append(
            {
                "pair_id": pair_id,
                "sentence_text": item["sentence_text"],
                "pmid": item["doc_id"].removeprefix("PMID:"),
                "entity_1": item["entity_1"]["surface_text"],
                "entity_2": item["entity_2"]["surface_text"],
                **dict.fromkeys(
                    [
                        "annotator_id",
                        "timestamp",
                        "decision",
                        "direction",
                        "independent_review",
                        "source",
                        "notes",
                    ],
                    "",
                ),
            }
        )
    if not rows:
        raise ValueError("Annotation queue is empty")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_reviews(path: str | Path | None) -> list[HumanReview]:
    """Read explicit human submissions from JSONL; absent submissions remain pending."""
    if path is None:
        return []
    if Path(path).suffix.lower() == ".csv":
        with Path(path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        records = []
        for row in rows:
            if not row.get("decision", "").strip():
                continue
            record = {key: value for key, value in row.items() if key in HumanReview.model_fields}
            if record.get("independent_review", "").lower() != "true":
                raise ValueError("CSV review must attest independent_review=true")
            record["independent_review"] = True
            records.append(HumanReview.model_validate(record))
        return records
    return [
        HumanReview.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def validation_report(queue_path: str | Path, reviews: list[HumanReview]) -> dict:
    """Report coverage and every reviewer pair's agreement, with no synthetic reviews."""
    queue = load_queue(queue_path)
    reviewers: dict[str, dict[str, HumanReview]] = {}
    enriched = []
    for review in reviews:
        if review.pair_id not in queue:
            raise ValueError(f"Review references unknown pair: {review.pair_id}")
        bucket = reviewers.setdefault(review.annotator_id, {})
        if review.pair_id in bucket:
            raise ValueError("Duplicate review by the same annotator for an item")
        bucket[review.pair_id] = review
        item = queue[review.pair_id]
        enriched.append(
            {
                **review.model_dump(mode="json"),
                "sentence_text": item["sentence_text"],
                "pmid": item["doc_id"].removeprefix("PMID:"),
            }
        )
    agreements = []
    from itertools import combinations

    for first, second in combinations(sorted(reviewers), 2):
        overlap = sorted(reviewers[first].keys() & reviewers[second].keys())
        a = [reviewers[first][p].decision for p in overlap]
        b = [reviewers[second][p].decision for p in overlap]
        # Kappa is undefined when expected agreement is one, even if all decisions agree.
        defined = bool(overlap) and len(set(a + b)) > 1
        agreements.append(
            {
                "reviewers": [first, second],
                "overlap": len(overlap),
                "kappa": float(cohen_kappa_score(a, b)) if defined else None,
                "raw_agreement": sum(x == y for x, y in zip(a, b, strict=False)) / len(a)
                if a
                else None,
                "disagreements": [
                    p
                    for p in overlap
                    if (reviewers[first][p].decision, reviewers[first][p].direction)
                    != (reviewers[second][p].decision, reviewers[second][p].direction)
                ],
            }
        )
    reviewed = {r.pair_id for r in reviews}
    double = {p for p in reviewed if sum(p in r for r in reviewers.values()) >= 2}
    return {
        "status": "pending" if not reviews else "awaiting_adjudication",
        "queue_size": len(queue),
        "reviews": enriched,
        "reviewed_items": len(reviewed),
        "double_reviewed_items": len(double),
        "reviewer_count": len(reviewers),
        "agreements": agreements,
        "human_gold_established": False,
        "independence": "Reviewer attestation; software cannot verify blinding or expertise",
        "pseudo_labels_included": False,
    }


class Adjudication(BaseModel):
    """Senior human decision after at least two independent reviews."""

    model_config = ConfigDict(extra="forbid")
    pair_id: str
    adjudicator_id: str
    timestamp: datetime
    decision: Literal["causal", "association", "none", "uncertain"]
    direction: Literal["entity_1->entity_2", "entity_2->entity_1", "not_applicable"]
    rationale: str
    source: Literal["human_adjudication"]

    @model_validator(mode="after")
    def validate_adjudication(self) -> "Adjudication":
        if (
            self.timestamp.tzinfo is None
            or not self.adjudicator_id.strip()
            or not self.rationale.strip()
        ):
            raise ValueError("Adjudication requires reviewer, timezone, and rationale")
        if (self.decision == "causal") != (self.direction != "not_applicable"):
            raise ValueError("Only causal adjudications require direction")
        return self


def adjudicate(
    queue_path: str | Path, reviews: list[HumanReview], decisions: list[Adjudication]
) -> dict:
    """Build a separate human gold export; never modify Phase 1 pseudo-label files."""
    report = validation_report(queue_path, reviews)
    queue = load_queue(queue_path)
    seen = set()
    gold = []
    for decision in decisions:
        if decision.pair_id not in queue or decision.pair_id in seen:
            raise ValueError("Unknown or duplicate adjudication pair")
        seen.add(decision.pair_id)
        matching = [r for r in reviews if r.pair_id == decision.pair_id]
        if len({r.annotator_id for r in matching}) < 2:
            raise ValueError("Adjudication requires two independent reviewers")
        if decision.adjudicator_id in {r.annotator_id for r in matching}:
            raise ValueError("Adjudicator must be separate from the independent reviewers")
        if any(decision.timestamp < r.timestamp for r in matching):
            raise ValueError("Adjudication cannot predate its reviews")
        if decision.decision != "uncertain":
            gold.append(
                {
                    **{
                        key: queue[decision.pair_id][key]
                        for key in [
                            "pair_id",
                            "doc_id",
                            "sent_id",
                            "sentence_text",
                            "entity_1",
                            "entity_2",
                            "split",
                        ]
                    },
                    "human_reviewed": True,
                    "validated_fields": ["relation_class", "direction"],
                    "review_status": "HUMAN_ADJUDICATED",
                    "labeling_method": "independent_human_review",
                    "relation_class": {"association": "association_only"}.get(
                        decision.decision, decision.decision
                    ),
                    "direction": decision.direction,
                    "adjudication": decision.model_dump(mode="json"),
                    "reviewer_ids": sorted({r.annotator_id for r in matching}),
                }
            )
    return {
        **report,
        "status": "adjudicated" if decisions else report["status"],
        "adjudicated_items": len(decisions),
        "gold_records": gold,
        "human_gold_established": len(gold) == len(queue) and bool(queue),
    }
