"""Canonical network construction and portable JSON model storage."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

from src.data.brfss import file_sha256
from src.mapping.registry import CanonicalRegistry


@dataclass
class FittedNetwork:
    model: DiscreteBayesianNetwork
    metadata: dict[str, Any]
    state_names: dict[str, list[str]]
    parent_counts: dict[str, list[float]] = field(default_factory=dict)


def build_network(graph_path: str | Path, config: dict) -> FittedNetwork:
    """Import the literature prior; all additional edges require exploratory configuration."""
    graph = nx.read_graphml(graph_path)
    registry = CanonicalRegistry().canonical_vars
    if not graph.is_directed() or not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Evidence graph must be a directed acyclic graph")
    if any(d.get("relation_class") != "causal" for _, _, d in graph.edges(data=True)):
        raise ValueError("Only causal-labelled evidence edges may initialize the network")
    nodes = config.get("variables", sorted(graph.nodes()))
    if not nodes or len(set(nodes)) != len(nodes) or any(n not in registry for n in nodes):
        raise ValueError("Model nodes must be unique canonical variables")
    if not set(graph.nodes()).issubset(nodes):
        raise ValueError("Model variables must include all imported prior nodes")
    mode = config.get("mode", "literature_prior")
    if mode not in {"literature_prior", "exploratory"}:
        raise ValueError("Unknown model mode")
    additions = config.get("exploratory_edges", [])
    if additions and mode != "exploratory":
        raise ValueError("Extra edges require an explicitly exploratory model")
    structure = nx.DiGraph()
    structure.add_nodes_from(nodes)
    structure.add_edges_from(graph.edges())
    for edge in additions:
        if not isinstance(edge, dict) or not edge.get("rationale", "").strip():
            raise ValueError("Exploratory edges require source, target, and rationale")
        source, target = edge["source"], edge["target"]
        if source not in nodes or target not in nodes or source == target:
            raise ValueError("Invalid exploratory parent-child relationship")
        if target in {"AgeCat", "Sex"}:
            raise ValueError("Baseline demographic variables cannot be child targets")
        structure.add_edge(source, target)
    if not nx.is_directed_acyclic_graph(structure):
        raise ValueError("Exploratory edges introduce a cycle")
    isolated = list(nx.isolates(structure))
    if isolated and not config.get("acknowledge_independence", False):
        raise ValueError(
            "Isolated variables assume independence; acknowledge_independence is required"
        )
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(nodes)
    model.add_edges_from(structure.edges())
    metadata = {
        "mode": mode,
        "structure_status": "project_accepted_prior_not_human_validated",
        "prior_path": str(graph_path),
        "prior_sha256": file_sha256(graph_path),
        "prior_edges": [list(e) for e in graph.edges()],
        "exploratory_edges": additions,
        "isolated_variables": isolated,
        "variable_definitions": {n: registry[n]["description"] for n in nodes},
        "human_validation": "pending",
        "causal_validity_established": False,
    }
    return FittedNetwork(
        model, metadata, {n: [str(s) for s in registry[n]["values"]] for n in nodes}
    )


def validate_cpds(bundle: FittedNetwork) -> dict:
    """Reject invalid or unnormalized CPDs and incompatible graph/cardinality schemas."""
    bundle.model.check_model()
    for cpd in bundle.model.get_cpds():
        values = cpd.get_values()
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or not np.allclose(values.sum(axis=0), 1, atol=1e-10)
        ):
            raise ValueError(f"Invalid CPD probabilities: {cpd.variable}")
        if cpd.state_names[cpd.variable] != bundle.state_names[cpd.variable]:
            raise ValueError("CPD states do not match registry schema")
    return {
        "normalized": True,
        "cpds": len(bundle.model.get_cpds()),
        "is_dag": nx.is_directed_acyclic_graph(bundle.model),
    }


def save_model(bundle: FittedNetwork, path: str | Path) -> None:
    """Persist numeric CPDs and metadata as JSON, never executable pickle."""
    validate_cpds(bundle)
    cpds = []
    for cpd in bundle.model.get_cpds():
        cpds.append(
            {
                "variable": cpd.variable,
                "parents": list(cpd.variables[1:]),
                "values": cpd.get_values().tolist(),
            }
        )
    payload = {
        "schema_version": 1,
        "nodes": list(bundle.model.nodes()),
        "edges": list(bundle.model.edges()),
        "state_names": bundle.state_names,
        "metadata": bundle.metadata,
        "parent_counts": bundle.parent_counts,
        "cpds": cpds,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, allow_nan=False))


def load_model(path: str | Path) -> FittedNetwork:
    """Restore and validate a portable fitted model."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported model schema")
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(data["nodes"])
    model.add_edges_from(data["edges"])
    states = data["state_names"]
    for c in data["cpds"]:
        parents = c["parents"]
        model.add_cpds(
            TabularCPD(
                c["variable"],
                len(states[c["variable"]]),
                c["values"],
                evidence=parents or None,
                evidence_card=[len(states[p]) for p in parents] or None,
                state_names={n: states[n] for n in [c["variable"], *parents]},
            )
        )
    bundle = FittedNetwork(model, data["metadata"], states, data["parent_counts"])
    validate_cpds(bundle)
    return bundle
