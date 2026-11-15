"""Controlled relation-class benchmark: rules, keyword, TF-IDF, and real MiniLM."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data.brfss import file_sha256
from src.evaluation.error_analysis import categorize_errors
from src.evaluation.relation import CLASSES, relation_metrics
from src.extractors.distilbert_extractor import KeywordHeuristicExtractor, insert_entity_markers
from src.extractors.rules_extractor import RulesExtractor
from src.schemas.evaluation import RelationEvaluation
from src.schemas.text import CandidatePair
from src.utils.config import load_yaml_config


def load_benchmark(path: str | Path) -> list[dict]:
    """Validate frozen document splits; never make a new convenient test set."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = set()
    documents: dict[str, str] = {}
    for record in records:
        if record["pair_id"] in seen or record["relation_class"] not in CLASSES:
            raise ValueError("Duplicate pair or invalid relation class in benchmark")
        seen.add(record["pair_id"])
        if record["split"] not in {"train", "dev", "test"}:
            raise ValueError("Every benchmark record needs a fixed train/dev/test split")
        if documents.setdefault(record["doc_id"], record["split"]) != record["split"]:
            raise ValueError("Document leakage between benchmark partitions")
    if {r["split"] for r in records} != {"train", "dev", "test"}:
        raise ValueError("All three benchmark partitions are required")
    return records


def classification_metrics(truth: list[str], predicted: list[str]) -> RelationEvaluation:
    """Class-only view of the common relation metric contract."""
    return relation_metrics(truth, predicted)


def pair_text(pair: CandidatePair) -> str:
    """Give both learned models the same explicit mention context and sentence."""
    marked = insert_entity_markers(
        pair.sentence_text,
        pair.entity_1.start_char,
        pair.entity_1.end_char,
        pair.entity_2.start_char,
        pair.entity_2.end_char,
    )
    return f"Entity one: {pair.entity_1.surface_text}. Entity two: {pair.entity_2.surface_text}. {marked}"


def benchmark(
    config_path: str | Path = "configs/benchmark.yaml",
    include_transformer: bool = True,
    output: str | Path | None = None,
) -> dict:
    """Train on train only; dev/test remain held out. No tuning on test or pseudo-gold claims."""
    config = load_yaml_config(config_path)
    records = load_benchmark(config["annotations"])
    pairs = [CandidatePair.model_validate(r) for r in records]
    texts = [pair_text(pair) for pair in pairs]
    y = np.array([r["relation_class"] for r in records])
    partitions = {
        split: np.array([i for i, r in enumerate(records) if r["split"] == split])
        for split in ["train", "dev", "test"]
    }
    train = partitions["train"]
    if len(set(y[train])) != 3:
        raise ValueError("Training must contain all three relation classes")
    predictions = {}
    detailed_predictions = {}
    timings: dict = {}
    for extractor in [RulesExtractor(), KeywordHeuristicExtractor()]:
        start = perf_counter()
        detailed_predictions[extractor.name] = extractor.predict(pairs)
        predictions[extractor.name] = np.array(
            [p.label.relation_class.value for p in detailed_predictions[extractor.name]]
        )
        timings[extractor.name] = {
            "inference_seconds": perf_counter() - start,
            "training_seconds": 0,
        }
    encoder = None
    truncated = None
    if include_transformer:
        from src.extractors.transformer_encoder import FrozenTransformerEncoder

        start = perf_counter()
        encoder = FrozenTransformerEncoder(config["encoder"])
        texts, truncated = encoder.common_inputs(texts)
        timings["encoder_setup_seconds"] = perf_counter() - start
    start = perf_counter()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, norm="l2", max_features=10000
    )
    x_train = vectorizer.fit_transform([texts[i] for i in train])
    clf_config = {**config["classifier"], "random_state": config["seed"]}
    tfidf_head = LogisticRegression(**clf_config).fit(x_train, y[train])
    train_time = perf_counter() - start
    start = perf_counter()
    predictions["TF-IDF + LogisticRegression"] = tfidf_head.predict(vectorizer.transform(texts))
    timings["TF-IDF + LogisticRegression"] = {
        "training_seconds": train_time,
        "inference_seconds": perf_counter() - start,
    }
    if encoder is not None:
        start = perf_counter()
        embeddings = encoder.encode(texts, Path("data/cache/benchmark_embeddings"))
        encode_time = perf_counter() - start
        start = perf_counter()
        transformer_head = LogisticRegression(**clf_config).fit(embeddings[train], y[train])
        train_time = perf_counter() - start
        start = perf_counter()
        predictions["MiniLM frozen encoder + LogisticRegression"] = transformer_head.predict(
            embeddings
        )
        timings["MiniLM frozen encoder + LogisticRegression"] = {
            "training_seconds": train_time,
            "inference_seconds": perf_counter() - start,
            "embedding_seconds_including_cache": encode_time,
        }
    tables = {
        split: {
            name: classification_metrics(y[index].tolist(), pred[index].tolist())
            for name, pred in predictions.items()
        }
        for split, index in partitions.items()
    }
    provisional = any(not r.get("human_reviewed", False) for r in records)
    report = {
        "annotations_path": str(config["annotations"]),
        "artifact_dir": str(output or config["output"]),
        "task": "relation_class_only",
        "label_status": "PROVISIONAL_RULE_PSEUDO_GOLD"
        if provisional
        else "human_reviewed_submissions",
        "human_benchmark_claim": False,
        "primary_metric": "macro_f1_supported_classes",
        "label_provenance": {
            "labeling_methods": sorted({r.get("labeling_method") or "unknown" for r in records}),
            "recorded_generator": "rule_heuristic_regex_v1"
            if provisional
            else "human_review_submissions",
            "generator_implementation_verified": False,
            "independent_gold": False,
            "rule_method_comparison_bias": "Rule-generated references may make related rule evaluations circular; exact generator implementation is not verified by metadata alone.",
        },
        "split_strategy": "existing_frozen_document_holdout",
        "input_sha256": file_sha256(config["annotations"]),
        "document_overlap": 0,
        "partitions": {
            s: {"pairs": len(i), "documents": len({records[j]["doc_id"] for j in i})}
            for s, i in partitions.items()
        },
        "split_pair_ids": {s: [records[j]["pair_id"] for j in i] for s, i in partitions.items()},
        "results": tables,
        "timings": timings,
        "encoder": encoder.metadata if encoder else {"status": "not_run_explicitly_disabled"},
        "common_tokenizer_truncated_examples": truncated,
        "same_classifier_settings": clf_config,
        "fairness": [
            "Same candidate IDs, document splits, relation classes, and metrics for every method",
            "TF-IDF vocabulary and both classifier heads fit on train only",
            "Both learned models receive the same token-bounded mention-marked texts",
            "Pretrained transformer is frozen; only its logistic head is fitted",
            "No test-based hyperparameter selection; settings fixed in configuration",
            "Rules operate on original sentence/mention spans; learned systems use bounded text",
            "Training time is not directly comparable to fixed pretrained-model inference",
        ],
        "limitations": [
            "Pseudo-labels are rule-derived, favouring related rules; not independent gold accuracy",
            "Dev/test have zero causal support: held-out causal generalization is unmeasured",
            "Small training set: only four causal training pairs",
            "Relation classification alone does not estimate direction/polarity or justify a DAG edge",
            "Historical LLM pilots cover different subsets and are excluded from this controlled comparison",
            "Frozen encoder benchmark is not full transformer fine-tuning; pretraining corpus overlap is unknown",
        ],
    }
    out = Path(output or config["output"])
    out.mkdir(parents=True, exist_ok=True)
    errors: dict = {}
    for split, indices in partitions.items():
        errors[split] = {}
        for name, predicted in predictions.items():
            failures = [
                {
                    "pair_id": records[i]["pair_id"],
                    "reference": str(y[i]),
                    "prediction": str(predicted[i]),
                    "sentence_text": records[i]["sentence_text"],
                }
                for i in indices
                if y[i] != predicted[i]
            ]
            item = {
                "evaluated_fields": ["relation_class"],
                "misclassified_count": len(failures),
                "misclassified_pairs": failures,
            }
            if name in detailed_predictions:
                item["full_extraction_diagnostics"] = categorize_errors(
                    [records[i] for i in indices], [detailed_predictions[name][i] for i in indices]
                )
                item["evaluated_fields"] = [
                    "relation_class",
                    "direction",
                    "polarity",
                    "language_status",
                ]
            errors[split][name] = item
    (out / "error_analysis.json").write_text(
        json.dumps(
            {
                "reference_status": report["label_status"],
                "primary_metric": report["primary_metric"],
                "partitions": errors,
            },
            indent=2,
            allow_nan=False,
        )
    )
    report["error_analysis_path"] = str(out / "error_analysis.json")
    (out / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    rows = []
    for split, models in tables.items():
        for name, metrics in models.items():
            rows.append(
                {
                    "split": split,
                    "model": name,
                    "n": metrics["n"],
                    "macro_f1_all_classes": metrics["macro_f1_all_three_classes"],
                    "macro_f1_supported_classes": metrics["macro_f1_supported_classes"],
                    "causal_support": metrics["per_class"]["causal"]["support"],
                    "causal_generalization_measured": metrics["causal_generalization_measured"],
                    "label_status": report["label_status"],
                }
            )
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    (out / "predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "pair_id": records[i]["pair_id"],
                    "split": records[i]["split"],
                    "predictions": {name: str(pred[i]) for name, pred in predictions.items()},
                }
            )
            + "\n"
            for i in range(len(records))
        )
    )
    # Portable learned heads, with the fitted TF-IDF vocabulary/IDF stored explicitly.
    heads = {
        "shared_tokenizer_applied": encoder is not None,
        "tfidf": {
            "classes": tfidf_head.classes_.tolist(),
            "coef": tfidf_head.coef_.tolist(),
            "intercept": tfidf_head.intercept_.tolist(),
            "vocabulary": vectorizer.vocabulary_,
            "idf": vectorizer.idf_.tolist(),
        },
    }
    if encoder is not None:
        heads["transformer"] = {
            "classes": transformer_head.classes_.tolist(),
            "coef": transformer_head.coef_.tolist(),
            "intercept": transformer_head.intercept_.tolist(),
            "encoder": encoder.metadata,
        }
    (out / "classifier_heads.json").write_text(json.dumps(heads, indent=2, default=int))
    return report
