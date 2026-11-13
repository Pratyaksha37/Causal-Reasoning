"""Portable trained relation-class heads for traditional and transformer features."""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.evaluation.benchmark import pair_text
from src.extractors.distilbert_extractor import KeywordHeuristicExtractor
from src.extractors.rules_extractor import RulesExtractor
from src.schemas.text import CandidatePair
from src.text.candidates import CandidateGenerator
from src.text.splitter import SentenceSplitter
from src.utils.config import load_yaml_config


def head_predictions(matrix: np.ndarray, head: dict) -> list[dict]:
    """Apply a saved multinomial logistic head without executable model serialization."""
    scores = matrix @ np.asarray(head["coef"]).T + np.asarray(head["intercept"])
    scores = np.asarray(scores)
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    classes = head["classes"]
    return [
        {
            "relation_class": classes[int(p.argmax())],
            "class_probabilities": dict(zip(classes, p.tolist(), strict=True)),
            "supported_fields": ["relation_class"],
        }
        for p in probabilities
    ]


def classify_pairs(
    pairs: list[CandidatePair],
    heads_path: str | Path = "results/benchmark/classifier_heads.json",
    config_path: str | Path = "configs/benchmark.yaml",
) -> dict[str, list[dict]]:
    """Predict using actual trained heads; never infer unsupported direction or polarity."""
    from src.extractors.transformer_encoder import FrozenTransformerEncoder

    if not pairs:
        return {}
    heads = json.loads(Path(heads_path).read_text())
    config = load_yaml_config(config_path)
    encoder = (
        FrozenTransformerEncoder(config["encoder"])
        if heads.get("shared_tokenizer_applied") or "transformer" in heads
        else None
    )
    if (
        "transformer" in heads
        and encoder is not None
        and encoder.metadata["weights_sha256"] != heads["transformer"]["encoder"]["weights_sha256"]
    ):
        raise ValueError("Saved transformer head does not match pretrained encoder weights")
    texts = [pair_text(p) for p in pairs]
    if heads.get("shared_tokenizer_applied", False) and encoder is not None:
        texts, _ = encoder.common_inputs(texts)
    head = heads["tfidf"]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, norm="l2", vocabulary=head["vocabulary"]
    )
    vectorizer.idf_ = np.asarray(head["idf"])
    predictions = {
        "TF-IDF + LogisticRegression": head_predictions(vectorizer.transform(texts), head)
    }
    if "transformer" in heads and encoder is not None:
        predictions["MiniLM frozen encoder + LogisticRegression"] = head_predictions(
            encoder.encode(texts), heads["transformer"]
        )
    return predictions


def extract_text(text: str, include_learned: bool = True) -> dict:
    """Run existing mention mapping/rules and optionally the trained comparison heads."""
    if not text.strip():
        raise ValueError("Text must not be empty")
    generator = CandidateGenerator()
    pairs: list[CandidatePair] = []
    for sentence in SentenceSplitter().split("USER_INPUT", text):
        pairs.extend(
            p
            for p in generator.generate_candidate_pairs(sentence)
            if p.entity_1.canonical_node_id != p.entity_2.canonical_node_id
        )
    results = {}
    for extractor in [RulesExtractor(), KeywordHeuristicExtractor()]:
        results[extractor.name] = [p.model_dump(mode="json") for p in extractor.predict(pairs)]
    if include_learned and pairs:
        results.update(classify_pairs(pairs))
    return {
        "input_kind": "user_supplied_text_not_verified_literature",
        "pairs": [p.model_dump(mode="json") for p in pairs],
        "predictions": results,
        "graph_updated": False,
        "limitations": "Learned heads classify relation only. A predicted causal word is not evidence of a causal effect.",
    }
