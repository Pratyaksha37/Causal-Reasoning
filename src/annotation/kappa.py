"""Inter-annotator agreement (Cohen's Kappa) and pre-adjudication disagreement report."""

import json
import logging
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)


def compute_agreement(
    annotator_1_records: list[dict[str, Any]],
    annotator_2_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute Cohen's Kappa and detailed disagreement report across two sets of annotations.

    Args:
        annotator_1_records: Records from Annotator 1.
        annotator_2_records: Records from Annotator 2 (double-annotated subset).

    Returns:
        Dict containing kappa scores, raw agreement rates, and disagreement details.
    """
    # Key by pair_id
    a1_dict = {r["pair_id"]: r for r in annotator_1_records if r.get("relation_class")}
    a2_dict = {r["pair_id"]: r for r in annotator_2_records if r.get("relation_class")}

    common_ids = sorted(set(a1_dict.keys()) & set(a2_dict.keys()))
    if not common_ids:
        logger.warning("No overlapping annotated pair_ids found between annotators.")
        return {
            "total_pairs_compared": 0,
            "relation_class_kappa": 0.0,
            "raw_agreement_rate": 0.0,
            "disagreements": [],
        }

    y1_rel: list[str] = []
    y2_rel: list[str] = []
    disagreements: list[dict[str, Any]] = []

    for pid in common_ids:
        r1 = a1_dict[pid]
        r2 = a2_dict[pid]

        c1 = r1["relation_class"].strip().lower()
        c2 = r2["relation_class"].strip().lower()

        y1_rel.append(c1)
        y2_rel.append(c2)

        if (
            c1 != c2
            or r1.get("direction") != r2.get("direction")
            or r1.get("polarity") != r2.get("polarity")
            or r1.get("language_status") != r2.get("language_status")
        ):
            disagreements.append(
                {
                    "pair_id": pid,
                    "sentence_text": r1.get("sentence_text"),
                    "entity_1": r1.get("entity_1_text"),
                    "entity_2": r1.get("entity_2_text"),
                    "annotator_1": {
                        "relation_class": c1,
                        "direction": r1.get("direction"),
                        "polarity": r1.get("polarity"),
                        "language_status": r1.get("language_status"),
                        "notes": r1.get("annotator_notes"),
                    },
                    "annotator_2": {
                        "relation_class": c2,
                        "direction": r2.get("direction"),
                        "polarity": r2.get("polarity"),
                        "language_status": r2.get("language_status"),
                        "notes": r2.get("annotator_notes"),
                    },
                }
            )

    # Cohen's Kappa for categorical relation_class
    kappa_val = float(cohen_kappa_score(y1_rel, y2_rel)) if len(set(y1_rel + y2_rel)) > 1 else 1.0
    raw_agreement = sum(1 for a, b in zip(y1_rel, y2_rel, strict=True) if a == b) / len(y1_rel)

    report = {
        "total_pairs_compared": len(common_ids),
        "relation_class_kappa": round(kappa_val, 4),
        "raw_agreement_rate": round(raw_agreement, 4),
        "total_disagreements": len(disagreements),
        "disagreements": disagreements,
    }

    logger.info(
        "Agreement computed on %d pairs: Kappa = %.4f, Raw Agreement = %.2f%% (%d disagreements)",
        len(common_ids),
        kappa_val,
        raw_agreement * 100.0,
        len(disagreements),
    )
    return report


def save_agreement_report(report: dict[str, Any], output_path: Path | str) -> None:
    """Save agreement report to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved agreement report to %s", path)
