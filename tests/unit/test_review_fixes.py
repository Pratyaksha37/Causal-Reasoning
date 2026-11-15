"""Regression coverage for reviewed correctness and reporting recommendations."""

import json
import subprocess
import sys
from pathlib import Path

from src.data.brfss import portable_source
from src.evaluation.benchmark import classification_metrics
from src.evaluation.metrics import evaluate_predictions
from src.schemas.prediction import Prediction
from src.utils.config import PROJECT_ROOT, load_yaml_config
from src.visualization.presentation import display_text, safe_evidence_url, safe_json


def test_keyword_heuristic_does_not_import_torch():
    code = """
import builtins
original = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError("Heuristic must not import torch")
    return original(name, *args, **kwargs)
builtins.__import__ = without_torch
from src.extractors.distilbert_extractor import KeywordHeuristicExtractor
assert not hasattr(KeywordHeuristicExtractor(), "device")
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_graph_orchestration_is_importable_outside_project(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.graph.pipeline import build_graph_from_saved_predictions; print(build_graph_from_saved_predictions.__module__)",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "src.graph.pipeline"
    config = load_yaml_config("configs/bayesian.yaml")
    assert config["paths"]["graph"] == "results/graph/current/evidence_graph_after_repair.graphml"
    assert config["paths"]["provenance"] == "results/graph/current/provenance.json"


def test_source_paths_are_portable(tmp_path):
    assert portable_source(PROJECT_ROOT / "data/brfss/raw/LLCP2015XPT.zip") == {
        "path": "data/brfss/raw/LLCP2015XPT.zip",
        "path_scope": "project_relative",
    }
    external = portable_source(tmp_path / "private/archive.zip")
    assert external == {"path": "archive.zip", "path_scope": "external_reference_only"}


def test_legacy_and_benchmark_share_class_metric_contract():
    truth = ["association_only", "none", "none"]
    predictions = ["none", "none", "none"]
    gold = [
        {"pair_id": str(i), "relation_class": label, "language_status": "affirmed"}
        for i, label in enumerate(truth)
    ]
    typed = [
        Prediction(
            pair_id=str(i),
            extractor_name="test",
            extractor_version="1",
            label={
                "relation_class": label,
                "direction": "not_applicable",
                "polarity": "not_applicable",
                "language_status": "affirmed",
            },
        )
        for i, label in enumerate(predictions)
    ]
    assert evaluate_predictions([], [])["primary_metric"] == "macro_f1_supported_classes"
    legacy = evaluate_predictions(gold, typed)
    benchmark = classification_metrics(truth, predictions)
    for key in [
        "primary_metric",
        "schema_version",
        "macro_f1_supported_classes",
        "macro_f1_all_three_classes",
        "confusion_matrix",
    ]:
        assert legacy[key] == benchmark[key]
    assert "direction" not in benchmark["evaluated_fields"]
    assert legacy["macro_f1"] == round(benchmark["macro_f1_all_three_classes"], 4)


def test_display_markup_and_embedded_script_payload_are_safe():
    text = "<h4>Methods</h4><b>Normal</b><script>alert(1)</script><img src=x onerror=alert(1)>"
    assert display_text(text) == "Methods Normal"
    hostile = {"sentence": '</script><script>alert("x")</script>&'}
    payload = safe_json(hostile)
    assert "<" not in payload and "&" not in payload
    assert json.loads(payload) == hostile
    assert safe_evidence_url("javascript:alert(1)") == "#"
    assert safe_evidence_url("https://europepmc.org/article/MED/123").startswith("https://")


def test_standalone_report_uses_safe_external_script(tmp_path):
    from src.modelling_pipeline import run
    from src.visualization.model_report import render_report

    summary = run(output=tmp_path / "fixture", demo=True)
    summary["comparisons"][0]["state"] = "</script><script>alert(1)</script>"
    render_report(summary, tmp_path)
    output = (tmp_path / "index.html").read_text()
    assert '<script src="viewer.js" defer>' in output
    assert "<script>const" not in output
    assert "</script><script>alert(1)" not in output
    assert (tmp_path / "viewer.js").exists()
    assert "root-treatment" in (tmp_path / "report.md").read_text()


def test_legacy_shell_flags_emit_notice_without_live_extraction(tmp_path):
    script = tmp_path / "run.sh"
    script.write_text(Path("run.sh").read_text())
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script), "--max-pairs", "100"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "Deprecated bare extraction flags" in result.stderr
    assert "run_pipeline.py" in result.stdout
