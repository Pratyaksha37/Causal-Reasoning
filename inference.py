"""Validated observational inference through pgmpy variable elimination."""

import numpy as np
from pgmpy.inference import VariableElimination

from src.bayesian.network import FittedNetwork, validate_cpds


def validate_assignments(bundle: FittedNetwork, assignments: dict[str, str]) -> None:
    for variable, value in assignments.items():
        if variable not in bundle.state_names:
            raise ValueError(
                f"Variable {variable!r} is not in this model; supported: {list(bundle.state_names)}"
            )
        if value not in bundle.state_names[variable]:
            raise ValueError(
                f"Invalid category {value!r} for {variable}: {bundle.state_names[variable]}"
            )


def query(
    bundle: FittedNetwork, target: str = "Outcome", evidence: dict[str, str] | None = None
) -> dict:
    """Compute P(target | evidence), rejecting impossible or invalid observations."""
    validate_cpds(bundle)
    evidence = evidence or {}
    if target not in bundle.state_names:
        raise ValueError(f"Unknown target: {target}")
    validate_assignments(bundle, evidence)
    if target in evidence:
        raise ValueError("Target must not also be an evidence variable")
    inference = VariableElimination(bundle.model)
    if evidence:
        joint = inference.query(list(evidence), show_progress=False)
        probability = float(joint.get_value(**evidence))
        if not np.isfinite(probability) or probability <= 0:
            raise ValueError("Evidence has zero probability in the model")
    factor = inference.query([target], evidence=evidence, show_progress=False)
    values = np.asarray(factor.values, dtype=float)
    if not np.isfinite(values).all() or not np.isclose(values.sum(), 1):
        raise ValueError("Inference produced an invalid distribution")
    return {
        "kind": "observational",
        "target": target,
        "evidence": evidence,
        "distribution": dict(zip(factor.state_names[target], values.tolist(), strict=False)),
        "model_mode": bundle.metadata.get("mode", "test_fixture"),
        "data_source": bundle.metadata.get("data_source", {"kind": "synthetic_test_fixture"}),
        "interpretation": "Conditional probability; not an intervention or clinical recommendation",
    }
