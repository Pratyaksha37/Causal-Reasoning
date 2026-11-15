"""Offline academic figures using the existing NetworkX/Matplotlib stack."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.bayesian.network import FittedNetwork


def render_figures(
    bundle: FittedNetwork, evaluation: dict, comparisons: list[dict], validation: dict, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5))
    display_graph = nx.DiGraph()
    display_graph.add_nodes_from(bundle.model.nodes())
    display_graph.add_edges_from(bundle.model.edges())
    nx.draw_networkx(
        display_graph,
        pos=nx.spring_layout(display_graph, seed=42),
        ax=ax,
        node_color="#dcecf0",
        edge_color="#264653",
        node_size=3500,
        arrowsize=25,
    )
    ax.set_title("Bayesian network · project-accepted structural prior")
    ax.text(
        0.5,
        -0.03,
        "Human validation pending · arrows do not establish causality",
        transform=ax.transAxes,
        ha="center",
    )
    ax.axis("off")
    fig.savefig(output / "network.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    cpds = bundle.model.get_cpds()
    fig, axes = plt.subplots(len(cpds), 1, figsize=(10, 3.5 * len(cpds)), squeeze=False)
    for ax, cpd in zip(axes[:, 0], cpds, strict=False):
        values = cpd.get_values()
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(cpd.state_names[cpd.variable])), cpd.state_names[cpd.variable])
        ax.set_title(f"P({cpd.variable} | {', '.join(cpd.variables[1:]) or 'no parents'})")
        from itertools import product

        labels = (
            [" / ".join(x) for x in product(*(cpd.state_names[p] for p in cpd.variables[1:]))]
            if len(cpd.variables) > 1
            else ["marginal"]
        )
        ax.set_xticks(range(values.shape[1]), labels, rotation=20)
        if values.size <= 100:
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    ax.text(
                        col,
                        row,
                        f"{values[row, col]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if values[row, col] > 0.6 else "black",
                    )
    fig.tight_layout()
    fig.savefig(output / "cpds.png", dpi=150)
    plt.close(fig)
    if comparisons:
        labels = [r["state"] for r in comparisons]
        observed = [r["observational"]["distribution"]["1"] for r in comparisons]
        intervened = [r["interventional"]["distribution"]["1"] for r in comparisons]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - 0.18, observed, 0.36, label="Observed BMI condition", color="#287a89")
        ax.bar(
            x + 0.18, intervened, 0.36, label="Hypothetical do(BMI), assumed graph", color="#dca75b"
        )
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability of prediabetes or diabetes")
        ax.legend()
        ax.set_title("Model comparison · causal validity is not established")
        fig.tight_layout()
        fig.savefig(output / "comparison.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    points = evaluation["calibration"]
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="Perfect calibration")
    ax.plot(
        [r["mean_predicted"] for r in points],
        [r["observed_rate"] for r in points],
        "o-",
        color="#287a89",
    )
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed outcome fraction",
        title="Held-out observational calibration",
    )
    fig.tight_layout()
    fig.savefig(output / "calibration.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        ["Queued", "Reviewed", "Double reviewed", "Adjudicated"],
        [
            validation["queue_size"],
            validation["reviewed_items"],
            validation["double_reviewed_items"],
            validation.get("adjudicated_items", 0),
        ],
        color="#287a89",
    )
    ax.set_title(f"Human evidence review · {validation['status']}")
    ax.set_ylabel("Candidate pairs")
    fig.tight_layout()
    fig.savefig(output / "validation.png", dpi=150)
    plt.close(fig)
