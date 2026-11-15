"""One relation-class calculation shared by legacy and integrated evaluators."""

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.schemas.evaluation import RelationClassMetrics, RelationEvaluation

CLASSES = ["causal", "association_only", "none"]


def relation_metrics(truth: list[str], predicted: list[str]) -> RelationEvaluation:
    """Expose unsupported classes explicitly; use supported-class macro F1 as primary."""
    if len(truth) != len(predicted) or not truth:
        raise ValueError("Predictions must cover every benchmark row exactly once")
    if (set(truth) | set(predicted)) - set(CLASSES):
        raise ValueError("Unknown relation class")
    p, r, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=CLASSES, zero_division=0
    )
    per_class: dict[str, RelationClassMetrics] = {}
    for i, name in enumerate(CLASSES):
        per_class[name] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "evaluable": bool(support[i]),
        }
        if support[i] == 0:
            per_class[name]["note"] = "Zero reference support; class generalization is unmeasured."
    return {
        "schema_version": 1,
        "n": len(truth),
        "evaluated_fields": ["relation_class"],
        "primary_metric": "macro_f1_supported_classes",
        "macro_f1_all_three_classes": float(f1.mean()),
        "macro_f1_supported_classes": float(f1[support > 0].mean()),
        "causal_generalization_measured": bool(support[0]),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=CLASSES).tolist(),
    }


def empty_relation_metrics() -> RelationEvaluation:
    """Keep the common schema present even when there is no evaluable input."""
    return {
        "schema_version": 1,
        "n": 0,
        "evaluated_fields": [],
        "primary_metric": "macro_f1_supported_classes",
        "macro_f1_all_three_classes": 0.0,
        "macro_f1_supported_classes": 0.0,
        "causal_generalization_measured": False,
        "per_class": {},
        "confusion_matrix": [[0, 0, 0] for _ in CLASSES],
    }
