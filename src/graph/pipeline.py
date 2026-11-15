"""Evidence graph reconstruction and export, independent of the root CLI."""

import logging
from pathlib import Path
from typing import Any

from src.graph.builder import EvidenceGraphBuilder
from src.graph.cycles import detect_cycles
from src.graph.exporter import export_association_graph, export_graph_for_phase2
from src.graph.repair import repair_graph_cycles
from src.graph.visualizer import plot_combined_evidence_graph, plot_evidence_graph
from src.schemas.prediction import Prediction
from src.schemas.text import CandidatePair
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def synthesize_and_export_graph(
    builder: EvidenceGraphBuilder,
    output_dir: Path | str = "results/graph",
    figures_dir: Path | str = "results/figures",
) -> dict[str, Any]:
    """Execute cycle detection, cycle repair, export to GraphML and provenance, and render plot."""
    causal_g = builder.causal_graph
    assoc_g = builder.association_graph
    logger.info(
        "Raw Graphs constructed: Causal DAG candidates: %d nodes, %d edges | Observational Associations: %d nodes, %d edges.",
        causal_g.number_of_nodes(),
        causal_g.number_of_edges(),
        assoc_g.number_of_nodes(),
        assoc_g.number_of_edges(),
    )

    # 1. Cycle Detection Before Repair on Causal Graph
    logger.info("--- Cycle Detection (Pre-Repair) ---")
    pre_cycles = detect_cycles(causal_g)
    logger.info(
        "Pre-repair cycle status: %d simple cycles, %d SCCs (DAG=%s)",
        pre_cycles["cycle_count"],
        pre_cycles["scc_count"],
        pre_cycles["is_dag"],
    )

    # 2. Deterministic Cycle Repair Heuristic
    logger.info("--- Executing Deterministic Cycle Repair ---")
    repaired_g, repair_log = repair_graph_cycles(causal_g)
    post_cycles = detect_cycles(repaired_g)
    logger.info(
        "Post-repair cycle status: %d simple cycles, %d SCCs (DAG=%s). Removed %d edges.",
        post_cycles["cycle_count"],
        post_cycles["scc_count"],
        post_cycles["is_dag"],
        len(repair_log),
    )

    # 3. Export for Phase 2
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    export_graph_for_phase2(
        graph=repaired_g,
        output_graphml=out_dir / "evidence_graph_after_repair.graphml",
        output_provenance_json=out_dir / "provenance.json",
        repair_log=repair_log,
        repair_log_file=out_dir / "cycle_repair.log",
        association_graph=assoc_g,
    )

    # Also export pre-repair causal graph for comparison
    export_graph_for_phase2(
        graph=causal_g,
        output_graphml=out_dir / "evidence_graph_before_repair.graphml",
        output_provenance_json=out_dir / "provenance_before_repair.json",
        association_graph=assoc_g,
    )

    # Export dedicated association graph
    export_association_graph(
        graph=assoc_g,
        output_graphml=out_dir / "association_graph.graphml",
    )

    # Export combined multi-relational graph
    combined_g = builder.get_combined_graph()
    export_graph_for_phase2(
        graph=combined_g,
        output_graphml=out_dir / "combined_evidence_graph.graphml",
        output_provenance_json=out_dir / "provenance_combined.json",
        association_graph=assoc_g,
    )

    # 4. Visualization
    logger.info("--- Rendering Graph Visualizations ---")
    plot_evidence_graph(repaired_g, fig_dir / "causal_dag.png")
    plot_combined_evidence_graph(
        causal_graph=repaired_g,
        association_graph=assoc_g,
        output_path=fig_dir / "evidence_graph_after_repair.png",
    )

    return {
        "raw_edges": causal_g.number_of_edges(),
        "pre_repair_cycles": pre_cycles["cycle_count"],
        "edges_removed": len(repair_log),
        "post_repair_edges": repaired_g.number_of_edges(),
        "is_dag": post_cycles["is_dag"],
        "filtering_stats": builder.get_filtering_stats(),
        "repaired_graph": repaired_g,
        "association_graph": assoc_g,
        "combined_graph": combined_g,
    }


def build_graph_from_saved_predictions(
    predictions_dir: Path | str = "results/predictions",
    corpus_candidates_file: Path | str = "data/candidates_corpus.jsonl",
    synthetic_benchmark_file: Path | str = "data/fixtures/synthetic_candidates.jsonl",
    include_synthetic: bool = False,
    output_dir: Path | str = "results/graph",
    figures_dir: Path | str = "results/figures",
) -> dict[str, Any]:
    """
    Offline graph building: Load existing saved predictions from results/predictions/,
    aggregate evidence, apply cycle repair, and regenerate graph artifacts.
    Makes ZERO LLM or external API calls.
    """
    setup_logging(level="INFO")
    logger.info("=" * 70)
    logger.info("OFFLINE EVIDENCE GRAPH CONSTRUCTION (FROM SAVED PREDICTIONS)")
    logger.info("=" * 70)

    pred_dir = Path(predictions_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(
            f"Cannot run in offline mode: prediction directory '{pred_dir}' does not exist. "
            "Please run inference first or specify a valid directory with --predictions-dir."
        )

    raw_pred_files = sorted(pred_dir.glob("*_predictions.jsonl"))
    pred_files = []
    for pf in raw_pred_files:
        if (
            pf.name == "finetuned-distilbert_predictions.jsonl"
            and (pred_dir / "keywordheuristic_predictions.jsonl").exists()
        ):
            logger.info(
                "Skipping '%s': duplicate legacy alias of 'keywordheuristic_predictions.jsonl'.",
                pf.name,
            )
            continue
        pred_files.append(pf)

    if not pred_files:
        raise FileNotFoundError(
            f"Cannot run in offline mode: no '*_predictions.jsonl' files found in '{pred_dir}'. "
            "Please run inference first or check your prediction directory."
        )

    # 1. Load Candidate Pair lookup from literature corpus
    pair_dict: dict[str, CandidatePair] = {}
    corpus_path = Path(corpus_candidates_file)
    if corpus_path.exists():
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    p = CandidatePair.model_validate_json(line)
                    pair_dict[p.pair_id] = p
        logger.info("Loaded %d corpus candidate pairs from %s.", len(pair_dict), corpus_path)
    else:
        logger.warning("Corpus candidates file %s not found.", corpus_path)

    if include_synthetic:
        synth_path = Path(synthetic_benchmark_file)
        if synth_path.exists():
            with open(synth_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        p = CandidatePair.model_validate_json(line)
                        pair_dict[p.pair_id] = p
            logger.info("Loaded synthetic benchmark candidate pairs from %s.", synth_path)

    # 2. Ingest predictions from disk
    builder = EvidenceGraphBuilder()
    total_preds_loaded = 0
    predictions_per_file: dict[str, int] = {}

    for pfile in pred_files:
        ext_preds: list[Prediction] = []
        with open(pfile, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        pred = Prediction.model_validate_json(line)
                        ext_preds.append(pred)
                    except Exception as err:
                        logger.warning("Could not parse prediction line in %s: %s", pfile.name, err)

        predictions_per_file[pfile.name] = len(ext_preds)
        total_preds_loaded += len(ext_preds)
        logger.info("Loaded %d predictions from '%s'.", len(ext_preds), pfile.name)

        for pred in ext_preds:
            pair = pair_dict.get(pred.pair_id)
            if not pair:
                continue
            # Synthetic evaluation pairs excluded from evidence graph by default
            if not include_synthetic and (
                pair.pair_id.startswith("SYN:") or pair.doc_id.startswith("DOC:SYN")
            ):
                continue
            builder.add_prediction(pred, pair)

    stats = builder.get_filtering_stats()
    logger.info(
        "Ingestion completed: %d self-loop candidate pairs filtered out.",
        stats["filtered_self_loops_count"],
    )

    result = synthesize_and_export_graph(builder, output_dir=output_dir, figures_dir=figures_dir)
    result["predictions_loaded"] = total_preds_loaded
    result["predictions_per_file"] = predictions_per_file
    return result
