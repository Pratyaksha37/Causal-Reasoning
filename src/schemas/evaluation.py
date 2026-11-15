"""Shared result contracts for class-only and richer relation extraction evaluations."""

from typing import NotRequired, TypedDict


class RelationClassMetrics(TypedDict):
    precision: float
    recall: float
    f1: float
    support: int
    evaluable: bool
    note: NotRequired[str]


class RelationEvaluation(TypedDict):
    schema_version: int
    n: int
    evaluated_fields: list[str]
    primary_metric: str
    macro_f1_all_three_classes: float
    macro_f1_supported_classes: float
    causal_generalization_measured: bool
    per_class: dict[str, RelationClassMetrics]
    confusion_matrix: list[list[int]]


class FullRelationEvaluation(RelationEvaluation, total=False):
    """Additional fields apply only to extractors that predict these dimensions."""

    direction_accuracy_conditional_on_causal: float
    causal_direction_evaluated_count: int
    polarity_accuracy: float
    language_status_accuracy: float
    total_evaluated: int
    gold_total: int
    predictions_total: int
    missing_predictions_count: int
    duplicate_predictions_count: int
    extra_predictions_count: int
    macro_f1: float
