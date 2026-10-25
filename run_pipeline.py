"""End-to-end pipeline runner for Phase 1: extraction, evaluation, graph building, and cycle repair."""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.error_analysis import categorize_errors
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.plots import plot_confusion_matrices
from src.evaluation.tables import generate_evaluation_artifacts
from src.extractors.distilbert_extractor import DistilBERTExtractor, KeywordHeuristicExtractor
from src.extractors.llm_extractor import LLMExtractor
from src.extractors.rules_extractor import RulesExtractor
from src.graph.builder import EvidenceGraphBuilder
from src.graph.cycles import detect_cycles
from src.graph.exporter import export_association_graph, export_graph_for_phase2
from src.graph.repair import repair_graph_cycles
from src.graph.visualizer import plot_combined_evidence_graph, plot_evidence_graph
from src.retrieval.corpus import CorpusManager
from src.schemas.prediction import Prediction
from src.schemas.text import CandidatePair
from src.utils.config import load_yaml_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

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

    raw_pred_files = sorted(list(pred_dir.glob("*_predictions.jsonl")))
    pred_files = []
    for pf in raw_pred_files:
        if pf.name == "finetuned-distilbert_predictions.jsonl" and (pred_dir / "keywordheuristic_predictions.jsonl").exists():
            logger.info("Skipping '%s': duplicate legacy alias of 'keywordheuristic_predictions.jsonl'.", pf.name)
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
            if not include_synthetic and (pair.pair_id.startswith("SYN:") or pair.doc_id.startswith("DOC:SYN")):
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


def run_full_pipeline(
    max_eval_pairs: int | str | None = None,
    sample_strategy: str = "all",
    include_synthetic: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end pipeline from frozen manifest to repaired evidence graph."""
    setup_logging(level="INFO")
    logger.info("=" * 70)
    logger.info("STARTING PHASE 1 CAUSAL RELATION EXTRACTION & EVIDENCE GRAPH PIPELINE")
    logger.info("=" * 70)

    limit: int | None = None
    if max_eval_pairs is not None:
        if isinstance(max_eval_pairs, str) and max_eval_pairs.lower() in ("all", "none"):
            limit = None
        else:
            limit = int(max_eval_pairs)

    # 1. Determinism
    exp_cfg = load_yaml_config("configs/experiment.yaml")
    set_seed(exp_cfg.get("seed", 42))

    # 2. Frozen Corpus Verification
    logger.info("--- Step 1: Loading Frozen Corpus Manifest ---")
    retrieval_cfg = load_yaml_config("configs/retrieval.yaml")
    corpus_manager = CorpusManager(retrieval_cfg)
    docs = corpus_manager.load_or_retrieve()
    logger.info("Verified %d abstracts from frozen manifest.", len(docs))

    # 3. Load Evaluation Pairs (Gold / Synthetic benchmark)
    logger.info("--- Step 2: Loading Evaluation Candidate Pairs ---")
    benchmark_file = Path("data/annotations/sample.jsonl")
    if not benchmark_file.exists():
        benchmark_file = Path("data/fixtures/synthetic_candidates.jsonl")
    gold_records: list[dict[str, Any]] = []
    with open(benchmark_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_records.append(json.loads(line))

    eval_pairs = [CandidatePair.model_validate(g) for g in gold_records]
    logger.info("Loaded %d gold evaluation candidate pairs from %s.", len(eval_pairs), benchmark_file)

    # Load corpus candidate pairs for graph construction
    corpus_pairs_file = Path("data/candidates_corpus.jsonl")
    all_non_self_pairs: list[CandidatePair] = []
    skipped_self_loops_count = 0
    if corpus_pairs_file.exists():
        with open(corpus_pairs_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = CandidatePair.model_validate_json(line)
                # Filter out self-loops during candidate selection so valid, diverse pairs are chosen
                if p.entity_1.canonical_node_id == p.entity_2.canonical_node_id:
                    skipped_self_loops_count += 1
                    continue
                all_non_self_pairs.append(p)

    # Determine candidate pairs based on sampling strategy and limit
    if limit is None or str(limit).lower() in ("all", "none") or sample_strategy == "all":
        sample_corpus_pairs = all_non_self_pairs
        logger.info(
            "Candidate pair selection: loaded all %d valid non-self-loop candidate pairs across full corpus (bypassed %d self-loops).",
            len(sample_corpus_pairs),
            skipped_self_loops_count,
        )
    elif sample_strategy == "random":
        import random
        rng = random.Random(exp_cfg.get("seed", 42))
        n_sample = min(int(limit), len(all_non_self_pairs))
        sample_corpus_pairs = rng.sample(all_non_self_pairs, n_sample)
        logger.info(
            "Candidate pair selection: sampled %d candidate pairs using seeded random strategy across full corpus.",
            len(sample_corpus_pairs),
        )
    elif sample_strategy == "stratified":
        import random
        from collections import defaultdict
        rng = random.Random(exp_cfg.get("seed", 42))
        grouped = defaultdict(list)
        for p in all_non_self_pairs:
            grouped[(p.entity_1.canonical_node_id, p.entity_2.canonical_node_id)].append(p)
        sample_corpus_pairs = []
        n_target = min(int(limit), len(all_non_self_pairs))
        for pair_key, group_items in grouped.items():
            k = max(1, int(round(len(group_items) / len(all_non_self_pairs) * n_target)))
            sample_corpus_pairs.extend(rng.sample(group_items, min(k, len(group_items))))
        if len(sample_corpus_pairs) > n_target:
            sample_corpus_pairs = sample_corpus_pairs[:n_target]
        logger.info(
            "Candidate pair selection: sampled %d candidate pairs using stratified strategy across %d distinct concept pairs.",
            len(sample_corpus_pairs),
            len(grouped),
        )
    else:  # sequential
        sample_corpus_pairs = all_non_self_pairs[: int(limit)]
        logger.info(
            "Candidate pair selection: sampled first %d candidate pairs sequentially by row order.",
            len(sample_corpus_pairs),
        )

    # Separate evaluation benchmark pairs from evidence graph candidates
    if include_synthetic:
        logger.info(
            "Flag --include-synthetic active: including %d synthetic evaluation pairs in graph candidates.",
            len(eval_pairs),
        )
        graph_pairs = eval_pairs + sample_corpus_pairs
    else:
        logger.info(
            "Building evidence graph solely from %d literature corpus candidate pairs (synthetic eval pairs excluded).",
            len(sample_corpus_pairs),
        )
        graph_pairs = sample_corpus_pairs

    # 4. Run the 3 Extractors
    logger.info("--- Step 3: Running the 3 Extractors ---")
    extractors = [
        RulesExtractor(),
        KeywordHeuristicExtractor(),
        LLMExtractor(),
    ]

    all_preds: dict[str, list[Prediction]] = {}
    eval_metrics: dict[str, dict[str, Any]] = {}
    error_summaries: dict[str, dict[str, Any]] = {}
    llm_cache_stats: dict[str, int] = {}

    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    for ext in extractors:
        logger.info("Running Extractor: %s (%s)...", ext.name, ext.version)
        # 1. Predict on eval set for benchmarking
        eval_preds = ext.predict(eval_pairs)
        # 2. Predict on literature graph candidate pairs
        graph_preds = ext.predict(graph_pairs) if graph_pairs else []
        all_preds[ext.name] = graph_preds

        # Save predictions
        pred_file = pred_dir / f"{ext.name.lower().replace(' ', '_')}_predictions.jsonl"
        with open(pred_file, "w", encoding="utf-8") as f:
            for p in graph_preds:
                f.write(p.model_dump_json() + "\n")

        # Evaluate against gold benchmark
        metrics = evaluate_predictions(gold_records, eval_preds)
        eval_metrics[ext.name] = metrics

        errors = categorize_errors(gold_records, eval_preds)
        error_summaries[ext.name] = errors

        if isinstance(ext, LLMExtractor):
            llm_cache_stats = ext.get_client_stats()
            logger.info(
                "LLM Client Cache Stats: %d hits, %d misses, %d API calls (%d failed)",
                llm_cache_stats.get("cache_hits", 0),
                llm_cache_stats.get("cache_misses", 0),
                llm_cache_stats.get("api_calls", 0),
                llm_cache_stats.get("failed_calls", 0),
            )

        logger.info(
            "[%s] Macro-F1: %.4f | Conditional Direction Acc: %.4f | Polarity Acc: %.4f",
            ext.name,
            metrics["macro_f1"],
            metrics["direction_accuracy_conditional_on_causal"],
            metrics["polarity_accuracy"],
        )

    # 5. Save Evaluation Summary Tables
    tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "intrinsic_evaluation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Extractor", "Macro-F1", "Causal-F1", "Assoc-F1", "None-F1",
            "Direction-Acc (Causal)", "Polarity-Acc", "Lang-Status-Acc"
        ])
        for name, m in eval_metrics.items():
            pc = m["per_class"]
            writer.writerow([
                name,
                m["macro_f1"],
                pc.get("causal", {}).get("f1", 0.0),
                pc.get("association_only", {}).get("f1", 0.0),
                pc.get("none", {}).get("f1", 0.0),
                m["direction_accuracy_conditional_on_causal"],
                m["polarity_accuracy"],
                m["language_status_accuracy"],
            ])
    logger.info("Saved evaluation table to %s", csv_path)

    # 6. Build Evidence Graph
    logger.info("--- Step 4: Constructing NetworkX Evidence Graph ---")
    builder = EvidenceGraphBuilder()
    for _ext_name, preds in all_preds.items():
        builder.build_from_pairs_and_predictions(graph_pairs, preds)

    # 7-10. Cycle detection, cycle repair, export, visualization
    graph_res = synthesize_and_export_graph(builder)

    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)

    summary_result = {
        "abstracts_loaded": len(docs),
        "evaluation_metrics": eval_metrics,
        "raw_edges": graph_res["raw_edges"],
        "pre_repair_cycles": graph_res["pre_repair_cycles"],
        "edges_removed": graph_res["edges_removed"],
        "post_repair_edges": graph_res["post_repair_edges"],
        "is_dag": graph_res["is_dag"],
        "filtering_stats": graph_res["filtering_stats"],
        "llm_cache_stats": llm_cache_stats,
    }
    return summary_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end causal relation extraction & evidence graph pipeline"
    )
    parser.add_argument(
        "--max-pairs",
        "--max-eval-pairs",
        dest="max_pairs",
        default="250",
        help="Max corpus candidate pairs to process in live pipeline (integer count or 'all')",
    )
    parser.add_argument(
        "--from-saved-predictions",
        action="store_true",
        help="Rebuild evidence graph offline from saved prediction files without running models/APIs",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="results/predictions",
        help="Directory containing saved prediction JSONL files for offline mode",
    )
    parser.add_argument(
        "--sample-strategy",
        choices=["all", "random", "stratified", "sequential"],
        default="all",
        help="Candidate sampling strategy: 'all' (process all 1,143 non-self pairs), 'random' (seeded uniform random), 'stratified' (proportional across concept pairs), or 'sequential' (file line order)",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic evaluation pairs in the evidence graph (disabled by default)",
    )
    args = parser.parse_args()

    # Parse max_pairs (integer or 'all')
    max_pairs_val: int | str | None = args.max_pairs
    if isinstance(args.max_pairs, str) and args.max_pairs.isdigit():
        max_pairs_val = int(args.max_pairs)

    if args.from_saved_predictions:
        res = build_graph_from_saved_predictions(
            predictions_dir=args.predictions_dir,
            include_synthetic=args.include_synthetic,
        )
        filt = res["filtering_stats"]
        print("\n" + "=" * 65)
        print("          PHASE 1 OFFLINE GRAPH RECONSTRUCTION SUMMARY")
        print("=" * 65)
        print(f"Predictions Processed:         {res['predictions_loaded']}")
        print(f"Filtered Self-Loops:           {filt['filtered_self_loops_count']}")
        print(f"Total Unique Evidence Nodes:   {filt['combined_nodes_count']}")
        print(f"Causal DAG Nodes / Edges:      {filt['causal_nodes_count']} nodes | {res['post_repair_edges']} edges (valid DAG={res['is_dag']})")
        print(f"Observational Assoc Edges:     {filt['association_nodes_count']} nodes | {filt['association_edges_count']} edges")
        print(f"Cycle Repair Edges Removed:    {res['edges_removed']} (pre-repair cycles={res['pre_repair_cycles']})")
        eval_res = generate_evaluation_artifacts(pred_dir=args.predictions_dir)
        plot_confusion_matrices(pred_dir=args.predictions_dir)
        print("-" * 65)
        print("Literature Benchmark Evaluation (N=250 Candidate Pairs):")
        for name, m in eval_res.get("literature_overall", {}).items():
            cf1 = m["per_class"].get("causal", {}).get("f1", 0.0)
            af1 = m["per_class"].get("association_only", {}).get("f1", 0.0)
            nf1 = m["per_class"].get("none", {}).get("f1", 0.0)
            print(
                f"  * {name:<22}: Macro-F1 = {m['macro_f1']:.4f} | Causal-F1 = {cf1:.4f} | Assoc-F1 = {af1:.4f} | None-F1 = {nf1:.4f}"
            )
        print("=" * 65)
    else:
        result = run_full_pipeline(
            max_eval_pairs=max_pairs_val,
            sample_strategy=args.sample_strategy,
            include_synthetic=args.include_synthetic,
        )
        filt = result["filtering_stats"]
        print("\n" + "=" * 65)
        print("               PHASE 1 EXECUTION SUMMARY TABLE")
        print("=" * 65)
        print(f"Corpus Abstracts Verified:     {result['abstracts_loaded']}")
        print(f"Total Unique Evidence Nodes:   {filt['combined_nodes_count']}")
        print(f"Causal DAG Nodes / Edges:      {filt['causal_nodes_count']} nodes | {result['post_repair_edges']} edges (valid DAG={result['is_dag']})")
        print(f"Observational Assoc Edges:     {filt['association_nodes_count']} nodes | {filt['association_edges_count']} edges")
        print(f"Cycle Repair Edges Removed:    {result['edges_removed']} (pre-repair cycles={result['pre_repair_cycles']})")
        print(f"Filtered Self-Loops:           {filt['filtered_self_loops_count']}")
        if result.get("llm_cache_stats"):
            c = result["llm_cache_stats"]
            print(f"LLM API / Cache Stats:         {c.get('cache_hits', 0)} hits, {c.get('cache_misses', 0)} misses, {c.get('api_calls', 0)} API calls")
        print("-" * 65)
        print("Extractor Performance (Macro-F1):")
        for name, m in result["evaluation_metrics"].items():
            print(
                f"  * {name:<22}: Macro-F1 = {m['macro_f1']:.4f} | Dir-Acc = {m['direction_accuracy_conditional_on_causal']:.4f}"
            )
        print("=" * 65)

    print("\nGenerated Artifacts:")
    print("  - Causal DAG (GraphML):         results/graph/evidence_graph_after_repair.graphml")
    print("  - Association Graph (GraphML):   results/graph/association_graph.graphml")
    print("  - Combined Graph (GraphML):      results/graph/combined_evidence_graph.graphml")
    print("  - Provenance JSON:              results/graph/provenance.json")
    print("  - Cycle Repair Log:             results/graph/cycle_repair.log")
    print("  - Evaluation Table (Overall):   results/tables/intrinsic_evaluation.csv")
    print("  - Evaluation Splits (Holdout):  results/tables/literature_evaluation_splits.csv")
    print("  - Diagnostic Error Analysis:    results/metrics/error_analysis.json")
    print("  - Confusion Matrix Plots:       results/figures/confusion_matrix_*.png")
    print("  - Causal DAG Plot:              results/figures/causal_dag.png")
    print("  - Combined Graph Visualization: results/figures/evidence_graph_after_repair.png\n")


if __name__ == "__main__":
    main()
