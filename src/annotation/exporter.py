"""Export sampled candidate pairs into tabular CSV and JSONL formats for human annotators."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def export_candidates_to_csv(
    input_sample: Path | str,
    output_csv: Path | str,
    annotator_name: str = "annotator_1",
) -> None:
    """
    Export sampled candidates to an intuitive CSV for human labelling.

    Args:
        input_sample: Path to sample.jsonl.
        output_csv: Target CSV file.
        annotator_name: Identifier for annotator column headers.
    """
    in_path = Path(input_sample)
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            e1 = item["entity_1"]
            e2 = item["entity_2"]

            # If annotator_2, only include items marked for double_annotate
            if annotator_name == "annotator_2" and not item.get("double_annotate", False):
                continue

            row = {
                "pair_id": item["pair_id"],
                "doc_id": item["doc_id"],
                "split": item["split"],
                "double_annotate": item.get("double_annotate", False),
                "sentence_text": item["sentence_text"],
                "entity_1_text": e1["surface_text"],
                "entity_1_span": f"{e1['start_char']}:{e1['end_char']}",
                "entity_1_canonical": e1["canonical_node_id"],
                "entity_2_text": e2["surface_text"],
                "entity_2_span": f"{e2['start_char']}:{e2['end_char']}",
                "entity_2_canonical": e2["canonical_node_id"],
                # Labels to fill:
                "relation_class": item.get("relation_class") or "",
                "direction": item.get("direction") or "",
                "polarity": item.get("polarity") or "",
                "language_status": item.get("language_status") or "",
                "annotator_notes": item.get("annotator_notes") or "",
            }
            rows.append(row)

    fieldnames = [
        "pair_id",
        "doc_id",
        "split",
        "double_annotate",
        "sentence_text",
        "entity_1_text",
        "entity_1_span",
        "entity_1_canonical",
        "entity_2_text",
        "entity_2_span",
        "entity_2_canonical",
        "relation_class",
        "direction",
        "polarity",
        "language_status",
        "annotator_notes",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Exported %d items for %s to %s", len(rows), annotator_name, out_path)
