"""Document-level candidate pair sampler with reproducible train/dev/test holdouts."""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

from src.retrieval.manifest import read_manifest
from src.schemas.text import CandidatePair
from src.text.candidates import CandidateGenerator
from src.text.splitter import SentenceSplitter
from src.utils.config import load_yaml_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def partition_documents(
    doc_ids: list[str],
    train_ratio: float = 0.60,
    dev_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Split document IDs into train, dev, and test sets.
    Preserves document boundaries to strictly prevent candidate pair leakage.
    """
    assert abs((train_ratio + dev_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    shuffled = sorted(set(doc_ids))
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_dev = int(n_total * dev_ratio)

    splits: dict[str, list[str]] = {
        "train": shuffled[:n_train],
        "dev": shuffled[n_train : n_train + n_dev],
        "test": shuffled[n_train + n_dev :],
    }
    return splits


def sample_candidates_for_annotation(
    manifest_path: str = "data/manifest/corpus_manifest.jsonl",
    config_path: str = "configs/experiment.yaml",
    output_sample: str = "data/annotations/sample.jsonl",
    output_splits: str = "data/annotations/doc_splits.json",
) -> list[dict[str, Any]]:
    """
    Sample candidate pairs stratified across document-level train/dev/test splits.
    Flags ~20% subset for double-annotation.
    """
    config = load_yaml_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    sampling_cfg = config.get("sampling", {})
    target_pairs = int(sampling_cfg.get("target_candidate_pairs", 250))
    train_ratio = float(sampling_cfg.get("train_ratio", 0.60))
    dev_ratio = float(sampling_cfg.get("dev_ratio", 0.20))
    test_ratio = float(sampling_cfg.get("test_ratio", 0.20))
    double_annotate_ratio = float(sampling_cfg.get("double_annotate_ratio", 0.20))

    docs = read_manifest(manifest_path, verify_hashes=True)
    all_doc_ids = [d.doc_id for d in docs]

    # 1. Document partition
    splits = partition_documents(
        doc_ids=all_doc_ids,
        train_ratio=train_ratio,
        dev_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    doc_to_split: dict[str, str] = {}
    for split_name, ids in splits.items():
        for doc_id in ids:
            doc_to_split[doc_id] = split_name

    # Save document splits for provenance and untouched test holdout
    splits_payload = {
        "seed": seed,
        "split_ratios": {"train": train_ratio, "dev": dev_ratio, "test": test_ratio},
        "counts": {k: len(v) for k, v in splits.items()},
        "splits": splits,
    }
    splits_path = Path(output_splits)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_payload, f, indent=2)
    logger.info("Saved document split assignments with seed %d to %s", seed, splits_path)

    # 2. Extract pairs from documents
    splitter = SentenceSplitter()
    generator = CandidateGenerator()

    pairs_by_split: dict[str, list[CandidatePair]] = {"train": [], "dev": [], "test": []}

    for doc in docs:
        split_name = doc_to_split.get(doc.doc_id, "train")
        sentences = splitter.split(doc.doc_id, doc.abstract_text)
        for sent in sentences:
            pairs = generator.generate_candidate_pairs(sent, split=split_name)
            # Filter self-loops so human annotators evaluate meaningful distinct entity pairs
            valid_pairs = [
                p for p in pairs if p.entity_1.canonical_node_id != p.entity_2.canonical_node_id
            ]
            pairs_by_split[split_name].extend(valid_pairs)

    # 3. Sample pairs proportionally
    rng = random.Random(seed)
    sampled_pairs: list[dict[str, Any]] = []

    split_targets = {
        "train": int(target_pairs * train_ratio),
        "dev": int(target_pairs * dev_ratio),
        "test": target_pairs - int(target_pairs * train_ratio) - int(target_pairs * dev_ratio),
    }

    for split_name, target in split_targets.items():
        available = pairs_by_split[split_name]
        chosen = (
            rng.sample(available, min(target, len(available)))
            if len(available) >= target
            else available
        )
        logger.info(
            "Sampled %d pairs for %s (available: %d)", len(chosen), split_name, len(available)
        )

        for p in chosen:
            # Deterministically select ~20% for double-annotation
            is_double = rng.random() < double_annotate_ratio
            record = {
                "pair_id": p.pair_id,
                "doc_id": p.doc_id,
                "sent_id": p.sent_id,
                "split": p.split,
                "sentence_text": p.sentence_text,
                "entity_1": p.entity_1.model_dump(),
                "entity_2": p.entity_2.model_dump(),
                "double_annotate": is_double,
                "review_status": "NEEDS_EXPERT_ANNOTATION",
                "human_reviewed": False,
                "is_llm_prelabeled": False,
                # Fields to be annotated by human:
                "relation_class": None,
                "direction": None,
                "polarity": None,
                "language_status": None,
                "annotator_notes": None,
            }
            sampled_pairs.append(record)

    # Save sampled candidate pairs
    sample_path = Path(output_sample)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_path, "w", encoding="utf-8") as f:
        for item in sampled_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(
        "Sampling complete: saved %d candidate pairs to %s (%d double-annotated)",
        len(sampled_pairs),
        sample_path,
        sum(1 for x in sampled_pairs if x["double_annotate"]),
    )
    return sampled_pairs


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Document-level candidate sampler for annotation")
    parser.add_argument("--manifest", default="data/manifest/corpus_manifest.jsonl")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--sample-out", default="data/annotations/sample.jsonl")
    parser.add_argument("--splits-out", default="data/annotations/doc_splits.json")
    args = parser.parse_args()

    sample_candidates_for_annotation(
        manifest_path=args.manifest,
        config_path=args.config,
        output_sample=args.sample_out,
        output_splits=args.splits_out,
    )


if __name__ == "__main__":
    main()
