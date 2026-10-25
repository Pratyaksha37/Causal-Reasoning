"""Unit tests for annotation export, import, and schema validation."""

import csv
from pathlib import Path

import pytest

from src.annotation.importer import import_annotations_from_csv


def test_import_valid_annotations(tmp_path: Path) -> None:
    csv_file = tmp_path / "valid_annotations.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "relation_class",
                "direction",
                "polarity",
                "language_status",
                "annotator_notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "P01",
                "relation_class": "causal",
                "direction": "entity_1->entity_2",
                "polarity": "increases",
                "language_status": "affirmed",
                "annotator_notes": "test note",
            }
        )
        writer.writerow(
            {
                "pair_id": "P02",
                "relation_class": "none",
                "direction": "not_applicable",
                "polarity": "not_applicable",
                "language_status": "affirmed",
                "annotator_notes": "",
            }
        )

    imported = import_annotations_from_csv(csv_file)
    assert len(imported) == 2
    assert imported[0]["label"]["relation_class"] == "causal"
    assert imported[0]["label"]["direction"] == "entity_1->entity_2"
    assert imported[1]["label"]["relation_class"] == "none"


def test_import_invalid_relation_class_raises(tmp_path: Path) -> None:
    csv_file = tmp_path / "invalid.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pair_id", "relation_class", "direction", "polarity", "language_status"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "P01",
                "relation_class": "unrecognized_relation_type",  # invalid
                "direction": "entity_1->entity_2",
                "polarity": "increases",
                "language_status": "affirmed",
            }
        )

    with pytest.raises(ValueError, match="Validation error"):
        import_annotations_from_csv(csv_file)
