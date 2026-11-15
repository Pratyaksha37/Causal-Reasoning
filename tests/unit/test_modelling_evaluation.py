import pandas as pd
import pytest

from src.bayesian.estimation import estimate_cpds
from src.bayesian.network import build_network
from src.data.brfss import synthetic_fixture
from src.evaluation.probabilistic import evaluate, split_data
from src.evaluation.sensitivity import sensitivity_analysis


def test_holdout_and_evaluation():
    data = synthetic_fixture(rows=200)
    train, test, report = split_data(data, 42, 0.2)
    again, _, _ = split_data(data, 42, 0.2)
    pd.testing.assert_frame_equal(train, again)
    assert report["overlap"] == 0
    structure = build_network("results/graph/current/evidence_graph_after_repair.graphml", {})
    fitted = estimate_cpds(structure, train)
    metrics = evaluate(fitted, test, float((train.Outcome == "1").mean()))
    assert 0 <= metrics["brier_score"] <= 1
    assert metrics["log_loss"] >= 0
    assert sum(r["count"] for r in metrics["calibration"]) == len(test)
    assert metrics["causal_accuracy"] is None
    experiments = sensitivity_analysis(fitted, train, test)
    assert len(experiments) == 4
    ablation = experiments[-1]
    assert ablation["structure"]["active_edges"] == []
    assert ablation["structure"]["source_prior_edges"] == [["BMICat", "Outcome"]]
    assert ablation["counts_reestimated"]
    assert metrics["calibration_coverage"]["populated_bins"] == len(metrics["calibration"])
    assert metrics["calibration_coverage"]["distinct_predicted_probabilities"] <= 4
    with pytest.raises(ValueError):
        split_data(data.head(2), 42, 0.2)
