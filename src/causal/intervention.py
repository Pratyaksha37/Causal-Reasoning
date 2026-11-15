"""Interventions by graph surgery and truncated factorization, with assumption gates."""

from itertools import product

import numpy as np
from pgmpy.factors.discrete import TabularCPD

from src.bayesian.inference import query, validate_assignments
from src.bayesian.network import FittedNetwork, validate_cpds

ASSUMPTIONS = ["assume_causal_sufficiency", "assume_consistency", "acknowledge_structural_prior"]


def identification_report(
    bundle: FittedNetwork, interventions: dict[str, str], assumptions: dict
) -> dict:
    """Assess identification within an asserted fully observed causal DAG, not from data alone."""
    validate_assignments(bundle, interventions)
    unmet = [name for name in ASSUMPTIONS if assumptions.get(name) is not True]
    if bundle.model.latents:
        unmet.append("fully_observed_graph_required")
    violations = []
    unverified = []
    for variable, state in interventions.items():
        cpd = bundle.model.get_cpds(variable)
        parents = list(cpd.variables[1:])
        totals = bundle.parent_counts.get(variable)
        if totals is None:
            unverified.append(variable)
            continue
        configurations = (
            list(product(*(bundle.state_names[p] for p in parents))) if parents else [()]
        )
        if len(totals) != len(configurations):
            raise ValueError("Stored parent counts do not match CPD shape")
        # Stored raw cell counts allow positivity checks even when Bayesian smoothing is used.
        raw = bundle.metadata.get("cell_counts", {}).get(variable)
        if raw is None:
            unverified.append(variable)
            continue
        counts = np.asarray(raw)[bundle.state_names[variable].index(state)]
        for cfg, total, count in zip(configurations, totals, counts, strict=False):
            if total <= 0 or count <= 0:
                violations.append(
                    {
                        "variable": variable,
                        "state": state,
                        "parents": dict(zip(parents, cfg, strict=False)),
                        "observed_count": float(count),
                    }
                )
    supported = not unmet and not violations and not unverified
    return {
        "status": "identified_under_asserted_dag_assumptions"
        if supported
        else "model_simulation_only",
        "identified_under_assumptions": supported,
        "real_world_causal_validity_established": False,
        "unmet_assumptions": unmet,
        "positivity_violations": violations,
        "positivity_unverified_variables": unverified,
        "method": "Truncated factorization in a fully observed causal DAG",
        "limitations": [
            "No causal discovery or proof of absent unmeasured confounding",
            "No front-door, latent-confounder ID algorithm, or transportability claim",
            "BRFSS is cross-sectional self-report; timing and intervention consistency are assumptions",
        ],
    }


def intervene(
    bundle: FittedNetwork,
    target: str,
    interventions: dict[str, str],
    evidence: dict[str, str] | None = None,
    assumptions: dict | None = None,
) -> dict:
    """Set treatment CPDs to point masses after cutting incoming arrows; preserve descendants."""
    if not interventions:
        raise ValueError("At least one intervention is required")
    validate_cpds(bundle)
    validate_assignments(bundle, interventions)
    evidence = evidence or {}
    validate_assignments(bundle, evidence)
    if set(evidence) & set(interventions):
        raise ValueError("A variable cannot be both observed evidence and an intervention")
    assessment = identification_report(bundle, interventions, assumptions or {})
    model = bundle.model.copy()
    for variable, state in interventions.items():
        model.remove_edges_from(list(model.in_edges(variable)))
        model.remove_cpds(model.get_cpds(variable))
        states = bundle.state_names[variable]
        probabilities = [[float(s == state)] for s in states]
        model.add_cpds(
            TabularCPD(variable, len(states), probabilities, state_names={variable: states})
        )
    mutilated = FittedNetwork(model, bundle.metadata, bundle.state_names, bundle.parent_counts)
    result = query(mutilated, target, evidence)
    return {
        **result,
        "kind": "interventional",
        "interventions": interventions,
        "identification": assessment,
        "interpretation": "Hypothetical intervention under the stated graph and assumptions; not clinical efficacy",
    }
