"""Evaluation metrics for causal and associational relation extraction."""

import logging
from typing import Any

from sklearn.metrics import f1_score, precision_score, recall_score

from src.schemas.prediction import Prediction

logger = logging.getLogger(__name__)


def evaluate_predictions(
    gold_labels: list[dict[str, Any]],
    predictions: list[Prediction],
    penalize_missing: bool = True,
) -> dict[str, Any]:
    """
    Compute comprehensive intrinsic relation extraction metrics.

    Args:
        gold_labels: List of gold records (with 'pair_id', 'relation_class', 'direction', 'polarity', etc.)
        predictions: List of model Prediction objects matching gold_labels.
        penalize_missing: If True, missing predictions for gold pairs are treated as 'none' (penalizing recall).
                         If False, metrics are computed only over common pair IDs.

    Returns:
        Dict with per-class P/R/F1, support counts, macro-F1, conditional direction accuracy,
        polarity accuracy, and integrity statistics (missing, duplicates, extra).
    """
    if not gold_labels:
        return {
            "total_evaluated": 0,
            "gold_total": 0,
            "macro_f1": 0.0,
            "causal_generalization_measured": False,
            "per_class": {},
        }

    # 1. Detect duplicate prediction IDs (preserve first seen)
    pred_map: dict[str, Prediction] = {}
    duplicate_pred_ids: list[str] = []
    for p in predictions:
        if p.pair_id in pred_map:
            duplicate_pred_ids.append(p.pair_id)
        else:
            pred_map[p.pair_id] = p

    if duplicate_pred_ids:
        logger.warning(
            "Detected %d duplicate prediction pair_ids. Using first occurrence for each.",
            len(duplicate_pred_ids),
        )

    # 2. Check coverage against gold labels
    gold_id_set = {g["pair_id"] for g in gold_labels}
    missing_gold_ids = [g["pair_id"] for g in gold_labels if g["pair_id"] not in pred_map]
    extra_pred_ids = [pid for pid in pred_map if pid not in gold_id_set]

    if missing_gold_ids:
        logger.warning(
            "Evaluation integrity: %d / %d gold pairs missing predictions.",
            len(missing_gold_ids),
            len(gold_labels),
        )

    # 3. Determine evaluation pairs
    if penalize_missing:
        eval_gold_records = gold_labels
    else:
        eval_gold_records = [g for g in gold_labels if g["pair_id"] in pred_map]

    if not eval_gold_records:
        return {
            "total_evaluated": 0,
            "gold_total": len(gold_labels),
            "macro_f1": 0.0,
            "causal_generalization_measured": False,
            "per_class": {},
            "missing_predictions_count": len(missing_gold_ids),
            "duplicate_predictions_count": len(duplicate_pred_ids),
            "extra_predictions_count": len(extra_pred_ids),
        }

    y_true_rel: list[str] = []
    y_pred_rel: list[str] = []

    # Direction conditional on both being causal
    causal_dir_matches = 0
    causal_dir_total = 0

    # Polarity accuracy
    polarity_matches = 0
    polarity_total = 0

    # Language status accuracy
    lang_matches = 0
    lang_total = 0

    for g in eval_gold_records:
        pid = g["pair_id"]
        g_rel = g.get("relation_class", "none").strip().lower()
        y_true_rel.append(g_rel)

        if pid in pred_map:
            p = pred_map[pid]
            p_rel = p.label.relation_class.value.lower()
            y_pred_rel.append(p_rel)

            # Direction conditional on causal
            if g_rel == "causal" and p_rel == "causal":
                causal_dir_total += 1
                if g.get("direction") == p.label.direction.value:
                    causal_dir_matches += 1

            # Polarity
            if g_rel != "none":
                polarity_total += 1
                if g.get("polarity") == p.label.polarity.value:
                    polarity_matches += 1

            # Language status
            lang_total += 1
            if g.get("language_status") == p.label.language_status.value:
                lang_matches += 1
        else:
            # Missing prediction penalized as non-detection
            y_pred_rel.append("none")
            if g_rel != "none":
                polarity_total += 1

    classes = ["causal", "association_only", "none"]
    causal_support = sum(1 for y in y_true_rel if y == "causal")

    report: dict[str, Any] = {
        "total_evaluated": len(eval_gold_records),
        "gold_total": len(gold_labels),
        "predictions_total": len(predictions),
        "missing_predictions_count": len(missing_gold_ids),
        "duplicate_predictions_count": len(duplicate_pred_ids),
        "extra_predictions_count": len(extra_pred_ids),
        "causal_generalization_measured": causal_support > 0,
        "macro_f1": round(float(f1_score(y_true_rel, y_pred_rel, labels=classes, average="macro", zero_division=0)), 4),
        "per_class": {},
        "direction_accuracy_conditional_on_causal": round(causal_dir_matches / causal_dir_total, 4) if causal_dir_total > 0 else 1.0,
        "causal_direction_evaluated_count": causal_dir_total,
        "polarity_accuracy": round(polarity_matches / polarity_total, 4) if polarity_total > 0 else 1.0,
        "language_status_accuracy": round(lang_matches / lang_total, 4) if lang_total > 0 else 1.0,
    }

    for c in classes:
        support = sum(1 for y in y_true_rel if y == c)
        p_score = precision_score(y_true_rel, y_pred_rel, labels=[c], average="micro", zero_division=0)
        r_score = recall_score(y_true_rel, y_pred_rel, labels=[c], average="micro", zero_division=0)
        f_score = f1_score(y_true_rel, y_pred_rel, labels=[c], average="micro", zero_division=0)
        c_report: dict[str, Any] = {
            "precision": round(float(p_score), 4),
            "recall": round(float(r_score), 4),
            "f1": round(float(f_score), 4),
            "support": support,
            "evaluable": support > 0,
        }
        if support == 0:
            c_report["note"] = "Zero gold support in this split; metric is unmeasured."
        report["per_class"][c] = c_report

    return report
