"""The one-project workflow and controlled transformer benchmark."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluation.benchmark import classification_metrics, load_benchmark
from src.extractors.learned_classifier import extract_text, head_predictions


def test_benchmark_rejects_document_leakage(tmp_path):
    records = load_benchmark("data/annotations/sample.jsonl")
    train_doc = next(r["doc_id"] for r in records if r["split"] == "train")
    next(r for r in records if r["split"] == "test")["doc_id"] = train_doc
    target = tmp_path / "leaked.jsonl"
    target.write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="Document leakage"):
        load_benchmark(target)


def test_classification_metrics_disclose_unsupported_causal_class():
    metrics = classification_metrics(["none", "association_only"], ["none", "association_only"])
    assert metrics["macro_f1_all_three_classes"] == pytest.approx(2 / 3)
    assert metrics["macro_f1_supported_classes"] == 1
    assert metrics["causal_generalization_measured"] is False
    with pytest.raises(ValueError):
        classification_metrics(["none"], [])


def test_portable_logistic_head():
    head = {"classes": ["a", "b", "c"], "coef": [[1, 0], [0, 1], [-1, -1]], "intercept": [0, 0, 0]}
    result = head_predictions(np.array([[5.0, 0.0]]), head)[0]
    assert result["relation_class"] == "a"
    assert sum(result["class_probabilities"].values()) == pytest.approx(1)
    assert result["supported_fields"] == ["relation_class"]


def test_arbitrary_text_extraction():
    result = extract_text("Obesity causes type 2 diabetes.", include_learned=False)
    assert result["pairs"]
    assert result["predictions"]["spaCy-Rules"][0]["label"]["relation_class"] == "causal"
    assert not result["graph_updated"]


def test_integrated_workflow_is_offline_and_isolated(tmp_path, monkeypatch):
    import requests

    from src.pipeline import run_project

    def forbid(*args, **kwargs):
        raise AssertionError("No network in offline project run")

    monkeypatch.setattr(requests.sessions.Session, "request", forbid)
    model_config = yaml.safe_load(Path("configs/bayesian.yaml").read_text())
    model_config["paths"].update(
        graph_output=str(tmp_path / "graph"), graph_figures=str(tmp_path / "graph_figures")
    )
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(model_config))
    benchmark_config = yaml.safe_load(Path("configs/benchmark.yaml").read_text())
    benchmark_config["output"] = str(tmp_path / "benchmark")
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(yaml.safe_dump(benchmark_config))
    result = run_project(
        model_path,
        demo=True,
        output=tmp_path / "model",
        report_root=tmp_path / "viewer",
        include_transformer=False,
        benchmark_config=benchmark_path,
    )
    assert result["corpus"]["hashes_verified"]
    assert result["nlp"]["encoder"]["status"] == "not_run_explicitly_disabled"
    assert result["bayesian"]["source"]["is_synthetic"]
    assert result["graph"]["is_dag"]
    assert (tmp_path / "viewer/index.html").exists()
    assert (tmp_path / "model/project_summary.json").exists()


def test_actual_transformer_benchmark_and_head_roundtrip(tmp_path, monkeypatch):
    if not Path("data/models/all-MiniLM-L6-v2/model.safetensors").exists():
        pytest.skip("Run prepare-encoder to execute the real pretrained-model integration test")
    from src.evaluation.benchmark import benchmark
    from src.extractors.learned_classifier import classify_pairs
    from src.schemas.text import CandidatePair

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    report = benchmark(output=tmp_path)
    assert report["encoder"]["parameter_count"] > 20_000_000
    assert report["encoder"]["encoder_finetuned"] is False
    for metrics in report["results"]["test"].values():
        assert metrics["n"] == 50
        assert not metrics["causal_generalization_measured"]
    assert report["document_overlap"] == 0
    assert report["primary_metric"] == "macro_f1_supported_classes"
    assert report["label_provenance"]["generator_implementation_verified"] is False
    errors = json.loads((tmp_path / "error_analysis.json").read_text())
    test_errors = errors["partitions"]["test"]
    assert "full_extraction_diagnostics" in test_errors["spaCy-Rules"]
    assert (
        "full_extraction_diagnostics"
        not in test_errors["MiniLM frozen encoder + LogisticRegression"]
    )
    for method, details in test_errors.items():
        matrix = report["results"]["test"][method]["confusion_matrix"]
        assert details["misclassified_count"] == sum(map(sum, matrix)) - sum(
            matrix[i][i] for i in range(3)
        )
    records = load_benchmark("data/annotations/sample.jsonl")[:2]
    restored = classify_pairs(
        [CandidatePair.model_validate(r) for r in records], tmp_path / "classifier_heads.json"
    )
    expected = [
        json.loads(line) for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ][:2]
    for name, predictions in restored.items():
        assert [p["relation_class"] for p in predictions] == [
            r["predictions"][name] for r in expected
        ]


def test_root_runner_exposes_integrated_commands():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "run_pipeline.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    for command in ["benchmark", "predict-text", "intervene", "prepare-encoder"]:
        assert command in result.stdout
