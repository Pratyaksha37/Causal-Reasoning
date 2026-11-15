"""Held-out prediction metrics measure observational fit, never causal accuracy."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from src.bayesian.inference import query
from src.bayesian.network import FittedNetwork


def split_data(
    data: pd.DataFrame, seed: int, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Deterministic row holdout; all CPD estimation occurs after this split."""
    if len(data) < 10 or not 0 < test_size < 1:
        raise ValueError("Need at least ten observations and test_size between zero and one")
    counts = data["Outcome"].value_counts()
    stratify = data["Outcome"] if len(counts) > 1 and counts.min() >= 2 else None
    train, test = train_test_split(data, test_size=test_size, random_state=seed, stratify=stratify)
    return (
        train,
        test,
        {
            "seed": seed,
            "test_size": test_size,
            "stratified": stratify is not None,
            "train_rows": len(train),
            "test_rows": len(test),
            "overlap": len(set(train.index) & set(test.index)),
        },
    )


def evaluate(bundle: FittedNetwork, test: pd.DataFrame, baseline_probability: float) -> dict:
    """Evaluate Outcome predictions using its observed Markov blanket, with grouped inference."""
    if "Outcome" not in bundle.model.nodes() or set(bundle.state_names["Outcome"]) != {"0", "1"}:
        raise ValueError("Evaluation requires canonical binary Outcome")
    if test.empty:
        raise ValueError("Test data cannot be empty")
    features = sorted(bundle.model.get_markov_blanket("Outcome"))
    cache: dict[tuple, float] = {}
    probabilities = []
    feature_rows = (
        test[features].itertuples(index=False, name=None) if features else [()] * len(test)
    )
    for row in feature_rows:
        if row not in cache:
            cache[row] = query(bundle, "Outcome", dict(zip(features, row, strict=False)))[
                "distribution"
            ]["1"]
        probabilities.append(cache[row])
    p = np.asarray(probabilities)
    y = test["Outcome"].astype(int).to_numpy()
    calibration = []
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=False):
        mask = (p >= low) & ((p < high) if high < 1 else (p <= high))
        if mask.any():
            calibration.append(
                {
                    "lower": float(low),
                    "upper": float(high),
                    "count": int(mask.sum()),
                    "mean_predicted": float(p[mask].mean()),
                    "observed_rate": float(y[mask].mean()),
                }
            )

    def score(probabilities: np.ndarray) -> dict:
        return {
            "log_loss": float(
                log_loss(y, np.column_stack([1 - probabilities, probabilities]), labels=[0, 1])
            ),
            "brier_score": float(brier_score_loss(y, probabilities)),
        }

    baseline = np.full(len(y), baseline_probability)
    return {
        "status": "observational_predictive_evaluation",
        "n_test": len(y),
        "conditioning_features": features,
        **score(p),
        "baseline": score(baseline),
        "baseline_training_prevalence": baseline_probability,
        "calibration": calibration,
        "calibration_coverage": {
            "populated_bins": len(calibration),
            "total_bins": 10,
            "distinct_predicted_probabilities": len(np.unique(p)),
            "minimum_prediction": float(p.min()),
            "maximum_prediction": float(p.max()),
            "limitation": "Only populated risk bins are assessed; calibration outside observed prediction support is unmeasured.",
        },
        "causal_accuracy": None,
        "causal_evaluation_requirements": "A prespecified causal estimand, defensible identification assumptions, and independent interventional or credible quasi-experimental outcome evidence; an RCT linkage alone does not define a generic causal accuracy metric.",
        "interpretation": "Unweighted held-out respondent prediction; not causal validation",
    }
