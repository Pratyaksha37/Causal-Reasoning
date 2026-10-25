"""Error analysis categorizer for relation extraction outputs."""

import logging
from collections import defaultdict
from typing import Any

from src.schemas.prediction import Prediction

logger = logging.getLogger(__name__)


def categorize_errors(
    gold_records: list[dict[str, Any]],
    predictions: list[Prediction],
) -> dict[str, Any]:
    """
    Categorize model errors into 5 specific diagnostic categories:
    1. direction_error
    2. assoc_vs_causal
    3. polarity_error
    4. negation_uncertainty_error
    5. entity_linking_error
    """
    pred_map = {p.pair_id: p for p in predictions}
    categorized: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for g in gold_records:
        pid = g["pair_id"]
        if pid not in pred_map:
            continue
        p = pred_map[pid]

        g_rel = g["relation_class"].strip().lower()
        p_rel = p.label.relation_class.value.lower()

        g_dir = g.get("direction")
        p_dir = p.label.direction.value

        g_pol = g.get("polarity")
        p_pol = p.label.polarity.value

        g_lang = g.get("language_status")
        p_lang = p.label.language_status.value

        error_entry = {
            "pair_id": pid,
            "sentence_text": g.get("sentence_text"),
            "entity_1": g.get("entity_1", {}).get("surface_text"),
            "entity_2": g.get("entity_2", {}).get("surface_text"),
            "gold": {"rel": g_rel, "dir": g_dir, "pol": g_pol, "lang": g_lang},
            "pred": {"rel": p_rel, "dir": p_dir, "pol": p_pol, "lang": p_lang},
        }

        # 1. Direction Error (Causal agreed, direction disagreed)
        if g_rel == "causal" and p_rel == "causal" and g_dir != p_dir:
            categorized["direction_error"].append(error_entry)

        # 2. Association vs Causation Error
        if (g_rel == "causal" and p_rel == "association_only") or (
            g_rel == "association_only" and p_rel == "causal"
        ):
            categorized["assoc_vs_causal"].append(error_entry)

        # 3. Polarity Error
        if g_rel == "causal" and p_rel == "causal" and g_pol != p_pol:
            categorized["polarity_error"].append(error_entry)

        # 4. Negation or Uncertainty Error
        if g_lang != p_lang:
            categorized["negation_uncertainty_error"].append(error_entry)

        # 5. Entity Linking Error
        e1_canon = g.get("entity_1", {}).get("canonical_node_id", "unmapped")
        e2_canon = g.get("entity_2", {}).get("canonical_node_id", "unmapped")
        if e1_canon == "unmapped" or e2_canon == "unmapped":
            categorized["entity_linking_error"].append(error_entry)

    summary = {
        "counts": {cat: len(items) for cat, items in categorized.items()},
        "examples_per_category": {
            cat: items[:2] for cat, items in categorized.items()
        },
    }
    return summary
