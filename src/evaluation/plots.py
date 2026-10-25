"""Confusion matrix and evaluation plots for relation extraction."""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from src.evaluation.tables import load_gold_records, load_predictions_from_dir
from src.schemas.prediction import Prediction

logger = logging.getLogger(__name__)


def plot_confusion_matrices(
    gold_path: Path | str = "data/annotations/sample.jsonl",
    pred_dir: Path | str = "results/predictions",
    figures_dir: Path | str = "results/figures",
) -> list[Path]:
    """Generate and save confusion matrix plots for each extractor on the gold benchmark."""
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    gold_records = load_gold_records(gold_path)
    if not gold_records:
        logger.warning("No gold records found at %s to plot confusion matrices.", gold_path)
        return []

    all_preds = load_predictions_from_dir(pred_dir)
    classes = ["causal", "association_only", "none"]
    class_labels = ["Causal", "Association", "None"]

    saved_plots: list[Path] = []

    for ext_name, preds in all_preds.items():
        pred_map = {p.pair_id: p for p in preds}
        y_true: list[str] = []
        y_pred: list[str] = []

        for g in gold_records:
            pid = g["pair_id"]
            if pid in pred_map:
                y_true.append(g["relation_class"].strip().lower())
                y_pred.append(pred_map[pid].label.relation_class.value.lower())

        if not y_true:
            continue

        cm = confusion_matrix(y_true, y_pred, labels=classes)

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(class_labels, fontsize=11, fontweight="medium")
        ax.set_yticklabels(class_labels, fontsize=11, fontweight="medium")

        ax.set_xlabel("Predicted Relation Class", fontsize=12, labelpad=8)
        ax.set_ylabel("Provisional Pseudo-Gold Label", fontsize=12, labelpad=8)
        ax.set_title(f"Confusion Matrix: {ext_name}\n(N={len(y_true)} Provisional Literature Pairs)", fontsize=13, fontweight="bold", pad=12)

        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.0
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                    fontweight="bold",
                )

        fig.tight_layout()
        out_path = fig_dir / f"confusion_matrix_{ext_name.lower().replace(' ', '_').replace('-', '_')}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        logger.info("Saved confusion matrix to %s", out_path)
        saved_plots.append(out_path)

    return saved_plots


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plot_confusion_matrices()
