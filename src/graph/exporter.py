"""Export evidence graphs to GraphML and JSON provenance formats for Phase 2."""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def export_graph_for_phase2(
    graph: nx.DiGraph | nx.Graph,
    output_graphml: Path | str,
    output_provenance_json: Path | str,
    repair_log: list[dict[str, Any]] | None = None,
    repair_log_file: Path | str | None = None,
    association_graph: nx.Graph | None = None,
) -> None:
    """
    Export evidence graph with full provenance consumable by Phase 2.
    Serializes complex attributes into strings for GraphML compliance.
    """
    g_export = graph.copy()

    provenance_data: dict[str, Any] = {
        "summary": {
            "causal_edges_count": graph.number_of_edges(),
            "causal_nodes_count": graph.number_of_nodes(),
            "association_edges_count": association_graph.number_of_edges() if association_graph else 0,
            "association_nodes_count": association_graph.number_of_nodes() if association_graph else 0,
        },
        "edges": [],  # Backwards compatibility: primary causal DAG edges
        "causal_edges": [],
        "association_edges": [],
    }

    for u, v, data in g_export.edges(data=True):
        prov_entry = {
            "source": u,
            "target": v,
            "relation_class": data.get("relation_class", "causal"),
            "polarity": data.get("polarity", "increases"),
            "support_count": data.get("support_count", 1),
            "unique_evidence_sentences": data.get("unique_evidence_sentences", data.get("support_count", 1)),
            "unique_papers_count": data.get("unique_papers_count", 0),
            "aggregated_confidence": data.get("aggregated_confidence", 1.0),
            "extractors": data.get("extractors", []),
            "evidence": data.get("evidence", []),
        }
        provenance_data["edges"].append(prov_entry)
        provenance_data["causal_edges"].append(prov_entry)

        # GraphML requires simple string or number attributes
        extractors_val = data.get("extractors", [])
        data["extractors"] = ",".join(extractors_val) if isinstance(extractors_val, list) else str(extractors_val)
        data["evidence_json"] = json.dumps(data.get("evidence", []))
        if "evidence" in data:
            del data["evidence"]

    if association_graph:
        for u, v, data in association_graph.edges(data=True):
            prov_entry = {
                "source": u,
                "target": v,
                "relation_class": "association_only",
                "polarity": data.get("polarity", "not_applicable"),
                "support_count": data.get("support_count", 1),
                "unique_evidence_sentences": data.get("unique_evidence_sentences", data.get("support_count", 1)),
                "unique_papers_count": data.get("unique_papers_count", 0),
                "aggregated_confidence": data.get("aggregated_confidence", 1.0),
                "extractors": data.get("extractors", []),
                "evidence": data.get("evidence", []),
            }
            provenance_data["association_edges"].append(prov_entry)

    # 1. Write GraphML
    graphml_path = Path(output_graphml)
    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g_export, graphml_path)
    logger.info("Saved GraphML to %s", graphml_path)

    # 2. Write Provenance JSON
    prov_path = Path(output_provenance_json)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(provenance_data, f, indent=2)
    logger.info("Saved provenance JSON to %s", prov_path)

    # 3. Write repair log if provided
    if repair_log is not None and repair_log_file:
        log_path = Path(repair_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            for item in repair_log:
                f.write(json.dumps(item) + "\n")
        logger.info("Saved cycle repair log to %s", log_path)


def export_association_graph(
    graph: nx.Graph,
    output_graphml: Path | str,
) -> None:
    """Export the observational association graph to GraphML format."""
    g_export = graph.copy()
    for _, _, data in g_export.edges(data=True):
        extractors_val = data.get("extractors", [])
        data["extractors"] = ",".join(extractors_val) if isinstance(extractors_val, list) else str(extractors_val)
        data["evidence_json"] = json.dumps(data.get("evidence", []))
        if "evidence" in data:
            del data["evidence"]

    graphml_path = Path(output_graphml)
    graphml_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(g_export, graphml_path)
    logger.info("Saved Association GraphML to %s", graphml_path)
