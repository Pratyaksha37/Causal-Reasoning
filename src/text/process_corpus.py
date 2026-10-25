"""Batch processing script to split frozen corpus, extract mentions, and generate candidate pairs."""

import argparse
import logging
from pathlib import Path

from src.mapping.alias_mapper import AliasMapper
from src.mapping.report import generate_mapping_report, save_mapping_report
from src.retrieval.manifest import read_manifest
from src.schemas.document import Document
from src.schemas.text import CandidatePair, Mention
from src.text.candidates import CandidateGenerator
from src.text.splitter import SentenceSplitter
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def process_corpus(
    manifest_path: str = "data/manifest/corpus_manifest.jsonl",
    output_candidates: str = "data/candidates_corpus.jsonl",
    output_report: str = "results/mapping_report.json",
) -> tuple[list[CandidatePair], list[Mention]]:
    """
    Process frozen corpus abstracts into sentences, mentions, and candidate pairs.
    """
    logger.info("Reading frozen corpus from %s", manifest_path)
    docs: list[Document] = read_manifest(manifest_path, verify_hashes=True)

    splitter = SentenceSplitter()
    alias_mapper = AliasMapper()
    generator = CandidateGenerator(alias_mapper=alias_mapper)

    all_mentions: list[Mention] = []
    all_pairs: list[CandidatePair] = []

    for doc in docs:
        sentences = splitter.split(doc.doc_id, doc.abstract_text)
        for sent in sentences:
            mentions = generator.extract_mentions(sent)
            all_mentions.extend(mentions)

            pairs = generator.generate_candidate_pairs(sent, mentions=mentions)
            all_pairs.extend(pairs)

    # Save mapping report
    report = generate_mapping_report(all_mentions)
    save_mapping_report(report, output_report)

    # Save candidate pairs to JSON Lines
    candidates_path = Path(output_candidates)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    with open(candidates_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(p.model_dump_json() + "\n")

    logger.info(
        "Corpus processed: %d abstracts -> %d mentions (%d mapped, %d unmapped) -> %d candidate pairs.",
        len(docs),
        len(all_mentions),
        report["mapped_count"],
        report["unmapped_count"],
        len(all_pairs),
    )
    return all_pairs, all_mentions


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Process frozen corpus into candidate pairs")
    parser.add_argument("--manifest", default="data/manifest/corpus_manifest.jsonl")
    parser.add_argument("--candidates-out", default="data/candidates_corpus.jsonl")
    parser.add_argument("--report-out", default="results/mapping_report.json")
    args = parser.parse_args()

    process_corpus(
        manifest_path=args.manifest,
        output_candidates=args.candidates_out,
        output_report=args.report_out,
    )


if __name__ == "__main__":
    main()
