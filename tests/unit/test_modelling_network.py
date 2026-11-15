import numpy as np
import pandas as pd
import pytest

from src.bayesian.estimation import estimate_cpds
from src.bayesian.network import build_network, load_model, save_model, validate_cpds

GRAPH = "results/graph/current/evidence_graph_after_repair.graphml"


def test_prior_and_roundtrip(tmp_path):
    bundle = build_network(GRAPH, {"variables": ["BMICat", "Outcome"]})
    assert set(bundle.model.edges()) == {("BMICat", "Outcome")}
    data = pd.DataFrame({"BMICat": ["normal"] * 4, "Outcome": ["0", "0", "0", "1"]})
    fit = estimate_cpds(bundle, data, equivalent_sample_size=8)
    # ESS/(2 outcomes * 4 BMI states) = one pseudocount per cell.
    np.testing.assert_allclose(fit.model.get_cpds("Outcome").get_values()[:, 1], [4 / 6, 2 / 6])
    np.testing.assert_allclose(fit.model.get_cpds("Outcome").get_values()[:, 0], [0.5, 0.5])
    assert validate_cpds(fit)["normalized"]
    save_model(fit, tmp_path / "model.json")
    restored = load_model(tmp_path / "model.json")
    for node in fit.model.nodes():
        np.testing.assert_array_equal(
            fit.model.get_cpds(node).get_values(), restored.model.get_cpds(node).get_values()
        )
    with pytest.raises(ValueError, match="Unobserved"):
        estimate_cpds(bundle, data, method="mle")


def test_reject_unsupported_and_cyclic_structure():
    with pytest.raises(ValueError, match="exploratory"):
        build_network(GRAPH, {"exploratory_edges": [{"source": "Outcome", "target": "BMICat"}]})
    with pytest.raises(ValueError, match="cycle"):
        build_network(
            GRAPH,
            {
                "mode": "exploratory",
                "exploratory_edges": [
                    {"source": "Outcome", "target": "BMICat", "rationale": "test only"}
                ],
            },
        )
    with pytest.raises(ValueError, match="independence"):
        build_network(GRAPH, {"variables": ["BMICat", "Outcome", "AgeCat"]})
