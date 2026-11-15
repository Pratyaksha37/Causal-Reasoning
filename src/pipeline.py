"""One project: corpus, NLP benchmark, evidence DAG, Bayesian model, and results viewer."""

import json
from pathlib import Path

from src.data.brfss import file_sha256
from src.evaluation.benchmark import benchmark
from src.modelling_pipeline import run as run_modelling
from src.retrieval.manifest import read_manifest
from src.utils.config import load_yaml_config


def run_project(
    config_path: str | Path = "configs/bayesian.yaml",
    data_path: str | Path | None = None,
    output: str | Path | None = None,
    demo: bool = False,
    reviews_path: str | Path | None = None,
    adjudications_path: str | Path | None = None,
    include_transformer: bool = True,
    benchmark_config: str | Path = "configs/benchmark.yaml",
    report_root: str | Path = "results",
) -> dict:
    """Demonstrate every stage with real saved evidence, common evaluation, and measured CPDs."""
    if not demo and data_path is None:
        raise ValueError("Supply --data with real BRFSS observations, or explicitly select --demo")
    config = load_yaml_config(config_path)
    out = Path(output or ("results/modelling/demo" if demo else config["paths"]["output"]))
    # Fail early on absent data or encoder rather than producing a partially fitted product.
    if not demo:
        assert data_path is not None
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Dataset unavailable: {data_path}")
    documents = read_manifest("data/manifest/corpus_manifest.jsonl", verify_hashes=True)
    comparison = benchmark(benchmark_config, include_transformer=include_transformer)
    from src.graph.pipeline import build_graph_from_saved_predictions

    graph_dir = Path(config["paths"].get("graph_output", "results/graph/current"))
    graph_figures = Path(config["paths"].get("graph_figures", "results/figures/graph"))
    graph = build_graph_from_saved_predictions(output_dir=graph_dir, figures_dir=graph_figures)
    # The rebuilt structure, not a separately maintained copy, initializes the probability model.
    resolved = {
        **config,
        "paths": {
            **config["paths"],
            "graph": str(graph_dir / "evidence_graph_after_repair.graphml"),
            "provenance": str(graph_dir / "provenance.json"),
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    import yaml

    resolved_path = out / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
    model = run_modelling(
        resolved_path.resolve(), data_path, out, demo, reviews_path, adjudications_path
    )
    project = {
        "artifacts": {
            "graph_dir": str(graph_dir),
            "graph_figures": str(graph_figures),
            "benchmark_dir": comparison["artifact_dir"],
        },
        "project": "Neuro-Symbolic Diabetes Evidence",
        "corpus": {
            "abstracts": len(documents),
            "hashes_verified": True,
            "manifest_sha256": file_sha256("data/manifest/corpus_manifest.jsonl"),
        },
        "nlp": comparison,
        "graph": {
            key: value
            for key, value in graph.items()
            if key not in {"repaired_graph", "association_graph", "combined_graph"}
        },
        "bayesian": model,
        "data_lineage": "Frozen biomedical text → saved sentence predictions → filtered prior DAG → BRFSS CPDs",
        "causal_validity_established": False,
    }
    (out / "project_summary.json").write_text(json.dumps(project, indent=2, allow_nan=False))
    from src.visualization.project_report import render_project_report

    render_project_report(project, out, Path(report_root))
    return project
