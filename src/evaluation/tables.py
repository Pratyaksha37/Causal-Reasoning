"""Results table generator and evaluator from saved raw predictions."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.evaluation.error_analysis import categorize_errors
from src.evaluation.metrics import evaluate_predictions
from src.schemas.prediction import Prediction

logger = logging.getLogger(__name__)


def load_gold_records(gold_path: Path | str) -> list[dict[str, Any]]:
    """Load gold candidate records from a JSONL file."""
    path = Path(gold_path)
    if not path.exists():
        logger.warning("Gold path %s does not exist.", path)
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_predictions_from_dir(pred_dir: Path | str) -> dict[str, list[Prediction]]:
    """Load all Prediction objects from a directory containing *_predictions.jsonl files."""
    p_dir = Path(pred_dir)
    preds: dict[str, list[Prediction]] = {}
    if not p_dir.exists():
        logger.warning("Predictions directory %s does not exist.", p_dir)
        return preds

    # Map filename prefixes to standard extractor display names
    display_names = {
        "spacy-rules_predictions.jsonl": "spaCy-Rules",
        "keywordheuristic_predictions.jsonl": "KeywordHeuristic",
        "prompted-llm-gemini_predictions.jsonl": "Prompted-LLM-Gemini",
        "prompted-llm-groq_predictions.jsonl": "Prompted-LLM-Groq",
    }

    for pfile in sorted(p_dir.glob("*_predictions.jsonl")):
        # Skip finetuned-distilbert_predictions.jsonl if keywordheuristic is present to prevent
        # presenting an identical duplicate file as an independent model.
        if pfile.name == "finetuned-distilbert_predictions.jsonl":
            if (p_dir / "keywordheuristic_predictions.jsonl").exists():
                logger.info(
                    "Skipping '%s': duplicate alias of 'KeywordHeuristic' (no trained Transformer exists).",
                    pfile.name,
                )
                continue
            ext_name = "KeywordHeuristic"
        else:
            ext_name = display_names.get(pfile.name, pfile.stem.replace("_predictions", ""))

        records: list[Prediction] = []
        with open(pfile, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(Prediction.model_validate_json(line))
                    except Exception as err:
                        logger.debug("Skipping invalid line in %s: %s", pfile.name, err)
        if records:
            preds[ext_name] = records
            logger.info("Loaded %d predictions for '%s' from %s", len(records), ext_name, pfile.name)

    return preds


def generate_evaluation_artifacts(
    gold_path: Path | str = "data/annotations/sample.jsonl",
    synthetic_gold_path: Path | str = "data/fixtures/synthetic_candidates.jsonl",
    pred_dir: Path | str = "results/predictions",
    tables_dir: Path | str = "results/tables",
    metrics_dir: Path | str = "results/metrics",
) -> dict[str, Any]:
    """
    Generate evaluation tables and error analysis from saved raw predictions.
    
    Generates:
      - results/tables/intrinsic_evaluation.csv (250 real literature candidates)
      - results/tables/literature_evaluation_splits.csv (train, dev, test breakdowns)
      - results/tables/synthetic_fixture_evaluation.csv (10 synthetic unit test fixtures)
      - results/metrics/error_analysis.json (diagnostic failure cases)
    """
    t_dir = Path(tables_dir)
    m_dir = Path(metrics_dir)
    t_dir.mkdir(parents=True, exist_ok=True)
    m_dir.mkdir(parents=True, exist_ok=True)

    gold_records = load_gold_records(gold_path)
    synth_records = load_gold_records(synthetic_gold_path)
    all_preds = load_predictions_from_dir(pred_dir)

    results: dict[str, Any] = {
        "literature_overall": {},
        "splits": {},
        "synthetic": {},
        "error_analysis": {},
    }

    # 1. Primary Evaluation: Real Literature Benchmark (250 pairs)
    if gold_records and all_preds:
        lit_csv_path = t_dir / "intrinsic_evaluation.csv"
        with open(lit_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Extractor", "Macro-F1", "Causal-F1", "Causal-Support", "Assoc-F1", "Assoc-Support",
                "None-F1", "None-Support", "Direction-Acc (Causal)", "Polarity-Acc", "Lang-Status-Acc",
                "Total-Evaluated", "Benchmark-Provenance"
            ])
            for ext_name, preds in all_preds.items():
                m = evaluate_predictions(gold_records, preds)
                if m.get("total_evaluated", 0) == 0:
                    continue
                results["literature_overall"][ext_name] = m
                pc = m["per_class"]
                writer.writerow([
                    ext_name,
                    m["macro_f1"],
                    pc.get("causal", {}).get("f1", 0.0),
                    pc.get("causal", {}).get("support", 0),
                    pc.get("association_only", {}).get("f1", 0.0),
                    pc.get("association_only", {}).get("support", 0),
                    pc.get("none", {}).get("f1", 0.0),
                    pc.get("none", {}).get("support", 0),
                    m["direction_accuracy_conditional_on_causal"],
                    m["polarity_accuracy"],
                    m["language_status_accuracy"],
                    m["total_evaluated"],
                    "PROVISIONAL_RULE_PSEUDO_GOLD",
                ])
        logger.info("Saved literature evaluation table to %s", lit_csv_path)

        # 2. Split-Wise Evaluation: Train, Dev, Test Holdout
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        for r in gold_records:
            s = r.get("split")
            if s in splits:
                splits[s].append(r)

        split_csv_path = t_dir / "literature_evaluation_splits.csv"
        with open(split_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Split", "Extractor", "Macro-F1", "Causal-F1", "Causal-Support", "Assoc-F1", "Assoc-Support",
                "None-F1", "None-Support", "Direction-Acc (Causal)", "Polarity-Acc", "Lang-Status-Acc",
                "Split-Support", "Causal-Generalization-Status"
            ])
            for split_name, split_recs in splits.items():
                results["splits"][split_name] = {}
                for ext_name, preds in all_preds.items():
                    m = evaluate_predictions(split_recs, preds)
                    if m.get("total_evaluated", 0) == 0:
                        continue
                    results["splits"][split_name][ext_name] = m
                    pc = m["per_class"]
                    causal_sup = pc.get("causal", {}).get("support", 0)
                    gen_status = "MEASURED" if causal_sup > 0 else "UNMEASURED (0 gold examples)"
                    writer.writerow([
                        split_name,
                        ext_name,
                        m["macro_f1"],
                        pc.get("causal", {}).get("f1", 0.0),
                        causal_sup,
                        pc.get("association_only", {}).get("f1", 0.0),
                        pc.get("association_only", {}).get("support", 0),
                        pc.get("none", {}).get("f1", 0.0),
                        pc.get("none", {}).get("support", 0),
                        m["direction_accuracy_conditional_on_causal"],
                        m["polarity_accuracy"],
                        m["language_status_accuracy"],
                        len(split_recs),
                        gen_status,
                    ])
        logger.info("Saved split-wise evaluation table to %s", split_csv_path)

        # 3. Diagnostic Error Analysis
        for ext_name, preds in all_preds.items():
            err_summary = categorize_errors(gold_records, preds)
            results["error_analysis"][ext_name] = err_summary

        error_json_path = m_dir / "error_analysis.json"
        with open(error_json_path, "w", encoding="utf-8") as f:
            json.dump(results["error_analysis"], f, indent=2)
        logger.info("Saved diagnostic error analysis to %s", error_json_path)

    # 4. Synthetic Fixture Evaluation (10 pairs)
    if synth_records and all_preds:
        synth_csv_path = t_dir / "synthetic_fixture_evaluation.csv"
        with open(synth_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Extractor", "Macro-F1", "Causal-F1", "Assoc-F1", "None-F1",
                "Direction-Acc (Causal)", "Polarity-Acc", "Lang-Status-Acc", "Total-Evaluated"
            ])
            for ext_name, preds in all_preds.items():
                m = evaluate_predictions(synth_records, preds)
                if m.get("total_evaluated", 0) == 0:
                    continue
                results["synthetic"][ext_name] = m
                pc = m["per_class"]
                writer.writerow([
                    ext_name,
                    m["macro_f1"],
                    pc.get("causal", {}).get("f1", 0.0),
                    pc.get("association_only", {}).get("f1", 0.0),
                    pc.get("none", {}).get("f1", 0.0),
                    m["direction_accuracy_conditional_on_causal"],
                    m["polarity_accuracy"],
                    m["language_status_accuracy"],
                    m["total_evaluated"],
                ])
        logger.info("Saved synthetic fixture evaluation table to %s", synth_csv_path)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_evaluation_artifacts()
