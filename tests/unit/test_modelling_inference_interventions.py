import pytest
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.bayesian.inference import query
from src.bayesian.network import FittedNetwork
from src.causal.intervention import intervene


def confounded_model():
    model = DiscreteBayesianNetwork([("C", "X"), ("C", "Y"), ("X", "Y")])
    states = {v: ["0", "1"] for v in model.nodes()}
    model.add_cpds(
        TabularCPD("C", 2, [[0.5], [0.5]], state_names={"C": states["C"]}),
        TabularCPD(
            "X",
            2,
            [[0.9, 0.1], [0.1, 0.9]],
            evidence=["C"],
            evidence_card=[2],
            state_names={v: states[v] for v in ["X", "C"]},
        ),
        TabularCPD(
            "Y",
            2,
            [[0.9, 0.8, 0.3, 0.2], [0.1, 0.2, 0.7, 0.8]],
            evidence=["C", "X"],
            evidence_card=[2, 2],
            state_names=states,
        ),
    )
    return FittedNetwork(
        model,
        {"mode": "synthetic_test", "cell_counts": {"X": [[90, 10], [10, 90]]}},
        states,
        {"X": [100, 100]},
    )


def test_confounding_changes_observation_vs_intervention():
    bundle = confounded_model()
    observed = query(bundle, "Y", {"X": "1"})
    simulated = intervene(bundle, "Y", {"X": "1"})
    assert observed["distribution"]["1"] == pytest.approx(0.74)
    assert simulated["distribution"]["1"] == pytest.approx(0.5)
    assert simulated["identification"]["status"] == "model_simulation_only"
    assert set(bundle.model.edges()) == {("C", "X"), ("C", "Y"), ("X", "Y")}
    assert query(bundle, "Y", {"X": "1"}) == observed
    assumptions = {
        "assume_causal_sufficiency": True,
        "assume_consistency": True,
        "acknowledge_structural_prior": True,
    }
    assert intervene(bundle, "Y", {"X": "1"}, assumptions=assumptions)["identification"][
        "identified_under_assumptions"
    ]
    assert intervene(bundle, "X", {"X": "0"})["distribution"]["0"] == 1
    assert intervene(bundle, "X", {"Y": "1"})["distribution"]["1"] == pytest.approx(0.5)


def test_invalid_query_and_positivity():
    bundle = confounded_model()
    for evidence in [{"X": "bad"}, {"unknown": "0"}, {"Y": "1"}]:
        with pytest.raises(ValueError):
            query(bundle, "Y", evidence)
    with pytest.raises(ValueError):
        intervene(bundle, "Y", {"X": "1"}, {"X": "0"})
    bundle.metadata["cell_counts"]["X"][1][0] = 0
    assert intervene(bundle, "Y", {"X": "1"})["identification"]["positivity_violations"]


def test_impossible_evidence():
    bundle = confounded_model()
    bundle.model.remove_cpds(bundle.model.get_cpds("C"))
    bundle.model.add_cpds(TabularCPD("C", 2, [[1], [0]], state_names={"C": ["0", "1"]}))
    with pytest.raises(ValueError, match="zero probability"):
        query(bundle, "Y", {"C": "1"})
