"""Unit tests for evaluation artifact generation, metrics tables, and plots."""

import json
from pathlib import Path

from src.evaluation.error_analysis import categorize_errors
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.plots import plot_confusion_matrices
from src.evaluation.tables import generate_evaluation_artifacts, load_gold_records
from src.schemas.prediction import (
    Direction,
    LanguageStatus,
    Polarity,
    Prediction,
    RelationClass,
    RelationLabel,
)


def test_evaluate_predictions_metrics_calculation() -> None:
    """Test standard P/R/F1, direction accuracy, and polarity accuracy."""
    gold = [
        {
            "pair_id": "P1",
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P2",
            "relation_class": "association_only",
            "direction": "not_applicable",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P3",
            "relation_class": "none",
            "direction": "not_applicable",
            "polarity": "not_applicable",
            "language_status": "affirmed",
        },
    ]

    preds = [
        Prediction(
            pair_id="P1",
            extractor_name="TestExt",
            extractor_version="1.0",
            label=RelationLabel(
                relation_class=RelationClass.CAUSAL,
                direction=Direction.E1_TO_E2,
                polarity=Polarity.INCREASES,
                language_status=LanguageStatus.AFFIRMED,
            ),
        ),
        Prediction(
            pair_id="P2",
            extractor_name="TestExt",
            extractor_version="1.0",
            label=RelationLabel(
                relation_class=RelationClass.ASSOCIATION_ONLY,
                direction=Direction.NOT_APPLICABLE,
                polarity=Polarity.INCREASES,
                language_status=LanguageStatus.AFFIRMED,
            ),
        ),
        Prediction(
            pair_id="P3",
            extractor_name="TestExt",
            extractor_version="1.0",
            label=RelationLabel(
                relation_class=RelationClass.NONE,
                direction=Direction.NOT_APPLICABLE,
                polarity=Polarity.NOT_APPLICABLE,
                language_status=LanguageStatus.AFFIRMED,
            ),
        ),
    ]

    m = evaluate_predictions(gold, preds)
    assert m["macro_f1"] == 1.0
    assert m["direction_accuracy_conditional_on_causal"] == 1.0
    assert m["polarity_accuracy"] == 1.0
    assert m["language_status_accuracy"] == 1.0


def test_categorize_errors_diagnostic_categories() -> None:
    """Verify that error analysis correctly isolates direction, assoc_vs_causal, and polarity errors."""
    gold = [
        {
            "pair_id": "P1",
            "sentence_text": "Sample text",
            "entity_1": {"surface_text": "BMI", "canonical_node_id": "BMICat"},
            "entity_2": {"surface_text": "T2DM", "canonical_node_id": "Outcome"},
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P2",
            "sentence_text": "Sample text 2",
            "entity_1": {"surface_text": "unknown", "canonical_node_id": "unmapped"},
            "entity_2": {"surface_text": "T2DM", "canonical_node_id": "Outcome"},
            "relation_class": "association_only",
            "direction": "not_applicable",
            "polarity": "increases",
            "language_status": "affirmed",
        },
    ]

    preds = [
        # Direction error
        Prediction(
            pair_id="P1",
            extractor_name="TestExt",
            extractor_version="1.0",
            label=RelationLabel(
                relation_class=RelationClass.CAUSAL,
                direction=Direction.E2_TO_E1,
                polarity=Polarity.INCREASES,
                language_status=LanguageStatus.AFFIRMED,
            ),
        ),
        # Causal vs Assoc error and entity linking error
        Prediction(
            pair_id="P2",
            extractor_name="TestExt",
            extractor_version="1.0",
            label=RelationLabel(
                relation_class=RelationClass.CAUSAL,
                direction=Direction.E1_TO_E2,
                polarity=Polarity.INCREASES,
                language_status=LanguageStatus.AFFIRMED,
            ),
        ),
    ]

    err = categorize_errors(gold, preds)
    assert "direction_error" in err["counts"]
    assert err["counts"]["direction_error"] == 1
    assert err["counts"]["assoc_vs_causal"] == 1
    assert err["counts"]["entity_linking_error"] == 1


def test_generate_evaluation_artifacts_end_to_end(tmp_path: Path) -> None:
    """Test generating evaluation tables and error analysis files without network calls."""
    gold_path = Path("data/annotations/sample.jsonl")
    pred_dir = Path("results/predictions")
    tables_dir = tmp_path / "tables"
    metrics_dir = tmp_path / "metrics"

    res = generate_evaluation_artifacts(
        gold_path=gold_path,
        pred_dir=pred_dir,
        tables_dir=tables_dir,
        metrics_dir=metrics_dir,
    )

    assert (tables_dir / "intrinsic_evaluation.csv").exists()
    assert (tables_dir / "literature_evaluation_splits.csv").exists()
    assert (metrics_dir / "error_analysis.json").exists()

    assert "spaCy-Rules" in res["literature_overall"]
    assert res["literature_overall"]["spaCy-Rules"]["macro_f1"] > 0.90
    assert "test" in res["splits"]
    assert len(res["splits"]["test"]) > 0
