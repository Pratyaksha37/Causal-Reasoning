"""Cycle detection and strongly connected components analysis for DiGraphs."""

from typing import Any

import networkx as nx


def detect_cycles(graph: nx.DiGraph) -> dict[str, Any]:
    """
    Detect and count simple cycles and strongly connected components (SCCs).

    Returns:
        Dict containing cycle_count, simple_cycles list, and scc_count (>1 node).
    """
    try:
        simple_cycles = list(nx.simple_cycles(graph))
    except Exception:
        simple_cycles = []

    sccs = [scc for scc in nx.strongly_connected_components(graph) if len(scc) > 1]

    return {
        "cycle_count": len(simple_cycles),
        "simple_cycles": simple_cycles,
        "scc_count": len(sccs),
        "sccs": [list(s) for s in sccs],
        "is_dag": nx.is_directed_acyclic_graph(graph),
    }
