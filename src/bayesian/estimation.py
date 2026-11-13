"""Count-based MLE and BDeu posterior-mean CPD estimation for pgmpy networks."""

from itertools import product

import numpy as np
import pandas as pd
from pgmpy.factors.discrete import TabularCPD

from src.bayesian.network import FittedNetwork, validate_cpds


def estimate_cpds(
    bundle: FittedNetwork,
    data: pd.DataFrame,
    method: str = "bayesian",
    equivalent_sample_size: float = 5.0,
) -> FittedNetwork:
    """Fit on training observations only; BDeu spreads ESS across all CPD cells."""
    if method not in {"mle", "bayesian"}:
        raise ValueError("Estimation method must be mle or bayesian")
    if method == "bayesian" and (
        not np.isfinite(equivalent_sample_size) or equivalent_sample_size <= 0
    ):
        raise ValueError("Equivalent sample size must be finite and positive")
    if data.empty or not set(bundle.model.nodes()).issubset(data.columns):
        raise ValueError("Nonempty training data must contain every model variable")
    for variable, allowed_states in bundle.state_names.items():
        if not data[variable].isin(allowed_states).all():
            raise ValueError(f"Missing or invalid training states for {variable}")
    # Work on a copy: a failed fit cannot corrupt a previously fitted model.
    model = bundle.model.copy()
    counts_by_node = {}
    cell_counts = {}
    for variable in model.nodes():
        parents = sorted(model.predecessors(variable))
        states = bundle.state_names
        configurations = list(product(*(states[p] for p in parents))) if parents else [()]
        r, q = len(states[variable]), len(configurations)
        if r * q > 1000000:
            raise ValueError("CPD is too large; reduce the exploratory parent set")
        grouped = data.groupby([variable, *parents], observed=True).size()
        counts = np.array(
            [
                [
                    float(grouped.get((state, *cfg) if parents else state, 0))
                    for cfg in configurations
                ]
                for state in states[variable]
            ]
        )
        cell_counts[variable] = counts.tolist()
        totals = counts.sum(axis=0)
        if method == "mle" and (totals == 0).any():
            raise ValueError(
                f"Unobserved parent configurations for {variable}; use Bayesian smoothing"
            )
        posterior = counts + (equivalent_sample_size / (r * q) if method == "bayesian" else 0)
        probabilities = posterior / posterior.sum(axis=0, keepdims=True)
        model.add_cpds(
            TabularCPD(
                variable,
                r,
                probabilities,
                evidence=parents or None,
                evidence_card=[len(states[p]) for p in parents] or None,
                state_names={n: states[n] for n in [variable, *parents]},
            )
        )
        counts_by_node[variable] = totals.tolist()
    fitted = FittedNetwork(
        model,
        {
            **bundle.metadata,
            "estimation": method,
            "equivalent_sample_size": equivalent_sample_size if method == "bayesian" else None,
            "cell_counts": cell_counts,
            "training_rows": len(data),
        },
        bundle.state_names,
        counts_by_node,
    )
    validate_cpds(fitted)
    return fitted
