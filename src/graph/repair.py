"""Deterministic cycle repair heuristic: IterativeMinConfidenceRemoval."""

import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


def repair_graph_cycles(graph: nx.DiGraph) -> tuple[nx.DiGraph, list[dict[str, Any]]]:
    """
    Repair cycles in a directed graph by iteratively removing the lowest-confidence edge.

    Tie-breaking:
    1. Lowest aggregated_confidence
    2. Lowest support_count (fewer literature mentions)
    3. Lexicographical order of (source, target)

    Returns:
        Tuple of (acyclic_graph, repair_log)
    """
    repaired_g = graph.copy()
    repair_log: list[dict[str, Any]] = []

    iteration = 0
    max_iterations = 1000

    while iteration < max_iterations:
        cycles = list(nx.simple_cycles(repaired_g))
        if not cycles:
            break

        # Sort cycles by length (shortest first)
        cycles.sort(key=len)
        cycle = cycles[0]

        # Extract all edges in the cycle: (u1, u2), (u2, u3), ..., (un, u1)
        cycle_edges: list[tuple[str, str]] = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            cycle_edges.append((u, v))

        # Sort candidate edges by (confidence ASC, support_count ASC, outcome_penalty DESC, (u, v) ASC)
        candidate_edges = []
        for u, v in cycle_edges:
            data = repaired_g[u][v]
            conf = data.get("aggregated_confidence", 1.0)
            sup = data.get("support_count", 1)
            # In disease epidemiology, the primary endpoint (Outcome: T2DM) does not cause
            # upstream demographic/lifestyle risk factors. Prioritize removing Outcome -> Exposure.
            outcome_as_cause = 1 if (u == "Outcome" and v != "Outcome") else 0
            candidate_edges.append(((conf, sup, -outcome_as_cause, u, v), u, v, data))

        candidate_edges.sort(key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4]))
        _, best_u, best_v, edge_data = candidate_edges[0]

        repaired_g.remove_edge(best_u, best_v)
        entry = {
            "iteration": iteration + 1,
            "removed_edge": f"{best_u}->{best_v}",
            "source": best_u,
            "target": best_v,
            "confidence": edge_data.get("aggregated_confidence"),
            "support_count": edge_data.get("support_count"),
            "cycle_detected": " -> ".join(cycle) + f" -> {cycle[0]}",
        }
        repair_log.append(entry)
        logger.info("Removed cycle edge: %s (confidence=%.4f, support=%d)", entry["removed_edge"], entry["confidence"], entry["support_count"])
        iteration += 1

    logger.info("Cycle repair complete. Removed %d edges. Final graph is DAG: %s", len(repair_log), nx.is_directed_acyclic_graph(repaired_g))
    return repaired_g, repair_log
