"""Sensitivity to smoothing and removing the sole literature prior edge."""

import pandas as pd

from src.bayesian.estimation import estimate_cpds
from src.bayesian.inference import query
from src.bayesian.network import FittedNetwork
from src.evaluation.probabilistic import evaluate


def sensitivity_analysis(
    structure: FittedNetwork, train: pd.DataFrame, test: pd.DataFrame
) -> list[dict]:
    """Refit only on training rows; ablation is explicitly exploratory."""
    baseline = float((train.Outcome == "1").mean())
    records = []
    for ess in [1.0, 5.0, 20.0]:
        fit = estimate_cpds(structure, train, equivalent_sample_size=ess)
        records.append(
            {
                "experiment": "smoothing",
                "equivalent_sample_size": ess,
                "evaluation": evaluate(fit, test, baseline),
                "bmi_risks": {
                    s: query(fit, evidence={"BMICat": s})["distribution"]["1"]
                    for s in fit.state_names.get("BMICat", [])
                },
            }
        )
    ablated = structure.model.copy()
    ablated.remove_edges_from(list(ablated.edges()))
    metadata = {
        key: value
        for key, value in structure.metadata.items()
        if key
        not in {
            "prior_edges",
            "cell_counts",
            "training_rows",
            "estimation",
            "equivalent_sample_size",
        }
    }
    metadata.update(
        mode="exploratory_edge_ablation",
        prior_edges=[],
        active_edges=[],
        source_prior_edges=structure.metadata.get("prior_edges", []),
    )
    # Counts and CPDs belong to the new parent sets. Do not inherit old counts;
    # estimate_cpds immediately recomputes them from the training observations.
    independent = FittedNetwork(ablated, metadata, structure.state_names, parent_counts={})
    fitted = estimate_cpds(independent, train)
    records.append(
        {
            "experiment": "exploratory_all_edges_removed_independence_assumption",
            "structure": {"active_edges": [], "source_prior_edges": metadata["source_prior_edges"]},
            "counts_reestimated": bool(fitted.parent_counts),
            "evaluation": evaluate(fitted, test, baseline),
        }
    )
    return records
