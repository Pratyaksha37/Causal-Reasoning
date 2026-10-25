"""Import and schema-validation of human annotations from CSV or JSONL."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.schemas.prediction import Direction, LanguageStatus, Polarity, RelationClass, RelationLabel

logger = logging.getLogger(__name__)


def import_annotations_from_csv(
    input_csv: Path | str,
    allow_unlabeled: bool = False,
) -> list[dict[str, Any]]:
    """
    Import and validate human annotations from CSV.

    Args:
        input_csv: Path to annotated CSV file.
        allow_unlabeled: If True, do not raise on empty label fields (for in-progress imports).

    Returns:
        List of validated annotation records.

    Raises:
        ValueError: If validation fails on non-empty rows.
    """
    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):
            pair_id = row.get("pair_id", "")
            rel_class = (row.get("relation_class") or "").strip().lower()
            direction = (row.get("direction") or "").strip().lower()
            polarity = (row.get("polarity") or "").strip().lower()
            lang_status = (row.get("language_status") or "").strip().lower()

            if not rel_class:
                if allow_unlabeled:
                    continue
                raise ValueError(f"Row {row_idx} ({pair_id}): 'relation_class' is required.")

            try:
                valid_rel = RelationClass(rel_class)
                valid_dir = (
                    Direction(direction)
                    if direction
                    else (Direction.NOT_APPLICABLE if valid_rel != RelationClass.CAUSAL else None)
                )
                if valid_dir is None:
                    raise ValueError(f"Direction required for causal relation on pair {pair_id}")

                valid_pol = (
                    Polarity(polarity)
                    if polarity
                    else (Polarity.NOT_APPLICABLE if valid_rel == RelationClass.NONE else None)
                )
                if valid_pol is None:
                    raise ValueError(f"Polarity required for pair {pair_id}")

                valid_lang = LanguageStatus(lang_status) if lang_status else LanguageStatus.AFFIRMED

                label = RelationLabel(
                    relation_class=valid_rel,
                    direction=valid_dir,
                    polarity=valid_pol,
                    language_status=valid_lang,
                    confidence=1.0,
                    reasoning=row.get("annotator_notes"),
                )
            except Exception as e:
                raise ValueError(f"Validation error on row {row_idx} (pair {pair_id}): {e}") from e

            row_data = dict(row)
            row_data["label"] = label.model_dump()
            records.append(row_data)

    logger.info("Successfully imported and validated %d annotated rows from %s", len(records), path)
    return records


def save_gold_labels_jsonl(
    records: list[dict[str, Any]],
    output_path: Path | str,
) -> None:
    """Save validated annotations to JSON Lines file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved %d validated gold records to %s", len(records), path)
