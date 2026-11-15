"""Offline-capable orchestration, isolated from all Phase 1 outputs."""

import importlib.metadata
import json
import platform
from pathlib import Path

import pandas as pd

from src.annotation.human_review import (
    Adjudication,
    adjudicate,
    export_review_form,
    read_reviews,
)
from src.bayesian.estimation import estimate_cpds
from src.bayesian.inference import query
from src.bayesian.network import build_network, save_model, validate_cpds
from src.causal.intervention import intervene
from src.data.brfss import file_sha256, load_data, portable_source, preprocess, synthetic_fixture
from src.evaluation.probabilistic import evaluate, split_data
from src.evaluation.sensitivity import sensitivity_analysis
from src.utils.config import load_yaml_config
from src.visualization.probability_plots import render_figures


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))


def run(
    config_path: str | Path = "configs/bayesian.yaml",
    data_path: str | Path | None = None,
    output: str | Path | None = None,
    demo: bool = False,
    reviews_path: str | Path | None = None,
    adjudications_path: str | Path | None = None,
) -> dict:
    """Fit, evaluate, save, and visualize; demo input is never substituted silently."""
    config = load_yaml_config(config_path)
    if demo and data_path is not None:
        raise ValueError("Choose real data or the explicit synthetic demo, not both")
    if not demo and data_path is None:
        raise ValueError("Supply --data with real BRFSS observations, or explicitly select --demo")
    out = Path(output or ("results/modelling/demo" if demo else config["paths"]["output"]))
    for protected in [
        "results/graph",
        "results/predictions",
        "results/tables",
        "data/annotations",
        "data/manifest",
    ]:
        if (
            out.resolve() == Path(protected).resolve()
            or Path(protected).resolve() in out.resolve().parents
        ):
            raise ValueError("Probabilistic modelling output cannot overwrite Phase 1 artifacts")
    out.mkdir(parents=True, exist_ok=True)
    structure = build_network(config["paths"]["graph"], config["model"])
    provenance_path = Path(config["paths"]["provenance"])
    structure.metadata["evidence_provenance"] = json.loads(provenance_path.read_text())
    reviews = read_reviews(reviews_path)
    decisions = (
        [
            Adjudication.model_validate_json(line)
            for line in Path(adjudications_path).read_text().splitlines()
            if line.strip()
        ]
        if adjudications_path
        else []
    )
    validation = adjudicate(config["paths"]["annotation_queue"], reviews, decisions)
    # Candidate review does not automatically certify the entire structural model.
    export_review_form(config["paths"]["annotation_queue"], out / "blank_review_form.csv")
    write_json(out / "validation.json", validation)
    if validation["gold_records"]:
        (out / "human_gold.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in validation["gold_records"])
        )
    preprocessing_config = dict(config["data"])
    if demo:
        preprocessing_config["profile"] = "canonical_test_fixture"
        raw = synthetic_fixture(config["seed"])
        source = {"kind": "synthetic_software_demo", "seed": config["seed"]}
    else:
        if preprocessing_config["profile"] == "canonical_test_fixture":
            raise ValueError(
                "Synthetic fixture profile requires --demo; real data cannot be relabelled"
            )
        assert data_path is not None
        raw = load_data(data_path, preprocessing_config["profile"])
        source = {
            "kind": "real_observations",
            **portable_source(data_path),
            "sha256": file_sha256(data_path),
        }
    data, preprocessing = preprocess(raw, preprocessing_config, list(structure.model.nodes()))
    source.update({"is_synthetic": demo})
    structure.metadata.update({"data_source": source, "unweighted": True})
    data.to_csv(out / "processed.csv", index=False)
    write_json(out / "preprocessing.json", preprocessing)
    train, test, split = split_data(data, config["seed"], config["test_size"])
    train_indices = set(train.index)
    pd.DataFrame(
        {
            "processed_row": data.index,
            "split": ["train" if i in train_indices else "test" for i in data.index],
        }
    ).to_csv(out / "splits.csv", index=False)
    # Save split lineage without retaining unrestricted input fields or outcome-derived predictors.
    estimation = config["estimation"]
    fitted = estimate_cpds(
        structure, train, estimation["method"], estimation["equivalent_sample_size"]
    )
    fitted.metadata["split"] = split
    save_model(fitted, out / "model.json")
    metrics = evaluate(fitted, test, float((train.Outcome == "1").mean()))
    comparisons = []
    if "BMICat" in fitted.model.nodes():
        for state in fitted.state_names["BMICat"]:
            comparisons.append(
                {
                    "state": state,
                    "observational": query(fitted, evidence={"BMICat": state}),
                    "interventional": intervene(
                        fitted, "Outcome", {"BMICat": state}, assumptions=config["causal"]
                    ),
                }
            )
    sensitivity = sensitivity_analysis(structure, train, test)
    summary = {
        "stage": "bayesian_modelling",
        "source": source,
        "model_fitted": True,
        "human_validation": validation["status"],
        "structure": {
            "nodes": list(fitted.model.nodes()),
            "edges": list(fitted.model.edges()),
            **validate_cpds(fitted),
            "status": fitted.metadata["structure_status"],
        },
        "preprocessing": preprocessing,
        "dataset_summary": {
            v: {str(k): int(n) for k, n in data[v].value_counts(dropna=False).items()} for v in data
        },
        "split": split,
        "evaluation": metrics,
        "comparisons": comparisons,
        "sensitivity": sensitivity,
        "causal_validity_established": False,
        "reproducibility": {
            "python": platform.python_version(),
            "config": config,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ["pgmpy", "pandas", "numpy", "networkx", "scikit-learn", "matplotlib"]
            },
        },
    }
    write_json(out / "summary.json", summary)
    write_json(out / "config.json", config)
    write_json(out / "sensitivity.json", sensitivity)
    write_json(out / "queries.json", comparisons)
    cpd_rows = []
    from itertools import product

    for cpd in fitted.model.get_cpds():
        parents = list(cpd.variables[1:])
        configurations = list(product(*(cpd.state_names[p] for p in parents))) if parents else [()]
        for col, cfg in enumerate(configurations):
            for row, state in enumerate(cpd.state_names[cpd.variable]):
                cpd_rows.append(
                    {
                        "variable": cpd.variable,
                        "parents": json.dumps(dict(zip(parents, cfg, strict=False))),
                        "state": state,
                        "probability": float(cpd.get_values()[row, col]),
                    }
                )
    pd.DataFrame(cpd_rows).to_csv(out / "cpds.csv", index=False)
    render_figures(fitted, metrics, comparisons, validation, out / "figures")
    from src.visualization.model_report import render_report

    render_report(summary, out)
    return summary
