"""Evidence graph visualization using NetworkX and Matplotlib."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import networkx as nx

logger = logging.getLogger(__name__)

PHASE2_CDC_VARS = {"AgeCat", "Sex", "Education", "Income", "Smoker", "PhysActivity", "BMICat", "Outcome"}


def plot_evidence_graph(
    graph: nx.DiGraph,
    output_path: Path | str,
    title: str = "Diabetes Literature Causal Evidence DAG (After Cycle Repair)",
) -> None:
    """
    Render and save a high-quality visualization of the causal DAG.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))

    if graph.number_of_nodes() == 0:
        plt.text(
            0.5,
            0.5,
            "No causal relationships extracted from candidate sample.\nRun with more candidates: ./run.sh --max-pairs 250\nor offline: ./run.sh --from-saved-predictions",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=13,
            fontweight="bold",
            color="#555555",
            transform=plt.gca().transAxes,
        )
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved empty evidence graph visualization to %s", path)
        return

    try:
        pos = nx.kamada_kawai_layout(graph)
    except Exception:
        pos = nx.spring_layout(graph, seed=42, k=2.2)

    node_colors = [
        "#4A90E2" if node in PHASE2_CDC_VARS else "#1ABC9C"
        for node in graph.nodes()
    ]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=3200,
        alpha=0.92,
        edgecolors="#2C3E50",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=10,
        font_weight="bold",
        font_color="#000000",
    )

    edge_colors = [
        "#E74C3C" if data.get("polarity") == "increases" else "#27AE60"
        for _, _, data in graph.edges(data=True)
    ]

    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=22,
        width=2.2,
        connectionstyle="arc3,rad=0.12",
    )

    edge_labels = {
        (u, v): f"{data.get('polarity', 'causes')}\n(n={data.get('support_count', 1)}, p={data.get('unique_papers_count', 1)})"
        for u, v, data in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8.5,
        label_pos=0.55,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#FADBD8", alpha=0.88, lw=0.6),
    )

    # Legend
    legend_handles = [
        mpatches.Patch(color="#4A90E2", label="CDC Phase 2 Variable"),
        mpatches.Patch(color="#1ABC9C", label="Open-Domain Concept"),
        mlines.Line2D([], [], color="#E74C3C", marker=">", markersize=8, label="Causal (Increases)"),
        mlines.Line2D([], [], color="#27AE60", marker=">", markersize=8, label="Causal (Decreases)"),
    ]
    plt.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=9.5)

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved evidence graph visualization to %s", path)


def plot_combined_evidence_graph(
    causal_graph: nx.DiGraph,
    association_graph: nx.Graph,
    output_path: Path | str,
    title: str = "Diabetes Literature Evidence Graph (Causal DAG & Observational Associations)",
) -> None:
    """
    Render and save an integrated multi-relational visualization:
    - Directed solid arrows for causal claims
    - Dashed lines for observational associations
    - Clean legend distinguishing node types and relation types
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build union graph for layout positioning
    union_g = nx.Graph()
    for n in causal_graph.nodes():
        union_g.add_node(n)
    for n in association_graph.nodes():
        union_g.add_node(n)
    for u, v in causal_graph.edges():
        union_g.add_edge(u, v)
    for u, v in association_graph.edges():
        union_g.add_edge(u, v)

    plt.figure(figsize=(15, 12))

    if union_g.number_of_nodes() == 0:
        plt.text(
            0.5,
            0.5,
            "No evidence relationships extracted from candidate sample.",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=13,
            fontweight="bold",
            color="#555555",
            transform=plt.gca().transAxes,
        )
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved empty combined evidence graph to %s", path)
        return

    try:
        pos = nx.kamada_kawai_layout(union_g)
    except Exception:
        pos = nx.spring_layout(union_g, seed=42, k=2.5, iterations=100)

    # 1. Draw Nodes
    node_colors = [
        "#4A90E2" if node in PHASE2_CDC_VARS else "#1ABC9C"
        for node in union_g.nodes()
    ]

    nx.draw_networkx_nodes(
        union_g,
        pos,
        node_color=node_colors,
        node_size=3600,
        alpha=0.92,
        edgecolors="#2C3E50",
        linewidths=1.8,
    )

    nx.draw_networkx_labels(
        union_g,
        pos,
        font_size=10,
        font_weight="bold",
        font_color="#000000",
    )

    # 2. Draw Association Edges (dashed, purple)
    # Suppress redundant association lines if an edge pair is already verified as causal
    if association_graph.number_of_edges() > 0:
        non_causal_assoc = [
            (u, v, d) for u, v, d in association_graph.edges(data=True)
            if not (causal_graph.has_edge(u, v) or causal_graph.has_edge(v, u))
        ]
        if non_causal_assoc:
            nx.draw_networkx_edges(
                association_graph,
                pos,
                edgelist=[(u, v) for u, v, _ in non_causal_assoc],
                edge_color="#8E44AD",
                style="dashed",
                width=1.8,
                alpha=0.75,
            )
            assoc_labels = {
                (u, v): f"assoc (n={d.get('support_count', 1)})"
                for u, v, d in non_causal_assoc
            }
            nx.draw_networkx_edge_labels(
                association_graph,
                pos,
                edge_labels=assoc_labels,
                font_size=7.5,
                font_color="#6C3483",
                label_pos=0.45,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#D2B4DE", alpha=0.85, lw=0.6),
            )

    # 3. Draw Causal Edges (solid arrows)
    if causal_graph.number_of_edges() > 0:
        causal_edges = list(causal_graph.edges(data=True))
        causal_colors = [
            "#E74C3C" if data.get("polarity") == "increases" else "#27AE60"
            for _, _, data in causal_edges
        ]
        nx.draw_networkx_edges(
            causal_graph,
            pos,
            edgelist=[(u, v) for u, v, _ in causal_edges],
            edge_color=causal_colors,
            arrows=True,
            arrowsize=24,
            width=2.5,
            connectionstyle="arc3,rad=0.12",
        )
        causal_labels = {
            (u, v): f"causal: {data.get('polarity', 'inc')}\n(n={data.get('support_count', 1)})"
            for u, v, data in causal_edges
        }
        nx.draw_networkx_edge_labels(
            causal_graph,
            pos,
            edge_labels=causal_labels,
            font_size=8.5,
            font_color="#900C3F",
            label_pos=0.58,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#FADBD8", alpha=0.88, lw=0.6),
        )

    # 4. Legend
    legend_handles = [
        mpatches.Patch(color="#4A90E2", label="CDC Phase 2 Variable"),
        mpatches.Patch(color="#1ABC9C", label="Open-Domain Concept"),
        mlines.Line2D([], [], color="#E74C3C", marker=">", markersize=8, label="Causal (Increases, Solid)"),
        mlines.Line2D([], [], color="#27AE60", marker=">", markersize=8, label="Causal (Decreases, Solid)"),
        mlines.Line2D([], [], color="#8E44AD", linestyle="--", linewidth=2, label="Observational Association (Dashed)"),
    ]
    plt.legend(handles=legend_handles, loc="upper right", framealpha=0.92, fontsize=9.5)

    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved combined evidence graph visualization to %s", path)
