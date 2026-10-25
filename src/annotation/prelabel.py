"""Optional LLM-assisted pre-annotation tooling with explicit non-gold flagging."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def flag_prelabeled_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Ensure all pre-labeled records are explicitly flagged to prevent accidental use as gold labels.
    """
    flagged: list[dict[str, Any]] = []
    for r in records:
        item = dict(r)
        item["is_llm_prelabeled"] = True
        item["human_reviewed"] = False
        item["review_status"] = "unreviewed"
        flagged.append(item)
    return flagged
