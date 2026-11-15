import hashlib
from pathlib import Path

import pytest

from src.bayesian.inference import query
from src.bayesian.network import load_model
from src.modelling_pipeline import run


def test_offline_end_to_end(tmp_path, monkeypatch):
    import requests

    def forbid(*args, **kwargs):
        raise AssertionError("Network access forbidden in offline test")

    monkeypatch.setattr(requests.sessions.Session, "request", forbid)
    prior = Path("results/graph/current/evidence_graph_after_repair.graphml")
    before = hashlib.sha256(prior.read_bytes()).hexdigest()
    result = run(output=tmp_path / "first", demo=True)
    repeated = run(output=tmp_path / "second", demo=True)
    assert result == repeated
    assert result["source"]["is_synthetic"]
    assert result["human_validation"] == "pending"
    assert not result["causal_validity_established"]
    assert result["structure"]["normalized"]
    assert hashlib.sha256(prior.read_bytes()).hexdigest() == before
    fitted = load_model(tmp_path / "first/model.json")
    assert sum(query(fitted)["distribution"].values()) == pytest.approx(1)
    for name in [
        "summary.json",
        "model.json",
        "cpds.csv",
        "index.html",
        "report.md",
        "splits.csv",
        "figures/comparison.png",
    ]:
        assert (tmp_path / "first" / name).stat().st_size > 0
    with pytest.raises(ValueError, match="Supply"):
        run(output=tmp_path / "missing")
