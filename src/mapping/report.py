"""Mapping analysis and reporting on mention frequencies and unmapped phrases."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from src.schemas.text import Mention

logger = logging.getLogger(__name__)


def generate_mapping_report(mentions: list[Mention]) -> dict[str, Any]:
    """
    Generate statistics on mapped vs unmapped mentions and frequent phrases.

    Args:
        mentions: List of all extracted Mention objects across the corpus.

    Returns:
        Dict containing counts, rates, breakdown per canonical concept, and top unmapped phrases.
    """
    total = len(mentions)
    if total == 0:
        return {
            "total_mentions": 0,
            "mapped_count": 0,
            "unmapped_count": 0,
            "mapped_rate": 0.0,
            "unmapped_rate": 0.0,
            "canonical_breakdown": {},
            "top_unmapped_phrases": [],
            "polarity_flipped_count": 0,
        }

    mapped_count = sum(1 for m in mentions if m.canonical_node_id != "unmapped")
    unmapped_count = total - mapped_count
    polarity_flipped_count = sum(1 for m in mentions if m.polarity_flip)

    concept_counter: Counter[str] = Counter()
    unmapped_counter: Counter[str] = Counter()

    for m in mentions:
        if m.canonical_node_id != "unmapped":
            concept_counter[m.canonical_node_id] += 1
        else:
            unmapped_counter[m.surface_text.lower().strip()] += 1

    report: dict[str, Any] = {
        "total_mentions": total,
        "mapped_count": mapped_count,
        "unmapped_count": unmapped_count,
        "mapped_rate": round(mapped_count / total, 4),
        "unmapped_rate": round(unmapped_count / total, 4),
        "polarity_flipped_count": polarity_flipped_count,
        "canonical_breakdown": dict(concept_counter.most_common()),
        "top_unmapped_phrases": [
            {"phrase": phrase, "count": count} for phrase, count in unmapped_counter.most_common(20)
        ],
    }

    return report


def save_mapping_report(report: dict[str, Any], output_path: Path | str) -> None:
    """Save mapping report to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved mapping report to %s", path)
