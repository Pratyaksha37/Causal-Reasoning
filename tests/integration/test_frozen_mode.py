"""Integration tests for frozen-mode corpus behavior and byte-identical reproducibility."""

from pathlib import Path

import pytest

from src.retrieval.corpus import CorpusManager, NetworkAccessForbiddenError
from src.retrieval.manifest import write_manifest
from src.schemas.document import Document


def test_frozen_mode_forbids_network_when_manifest_missing(tmp_path: Path) -> None:
    config = {
        "frozen_mode": True,
        "query": "test",
        "manifest": {
            "output_file": str(tmp_path / "missing_manifest.jsonl"),
            "exclusions_file": str(tmp_path / "missing_exclusions.jsonl"),
        },
    }
    manager = CorpusManager(config)
    with pytest.raises(
        FileNotFoundError, match="Frozen mode is active but manifest file does not exist"
    ):
        manager.load_or_retrieve()


def test_frozen_mode_byte_identical_reproducibility(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus_manifest.jsonl"
    docs = [
        Document.create(
            title=f"Study {i}",
            abstract_text=f"Abstract text for study number {i} with deterministic content.",
            pub_date="2024",
            query_provenance="frozen_test_query",
            retrieval_date="2026-10-22",
            pmid=f"1000{i}",
        )
        for i in range(5)
    ]
    write_manifest(docs, manifest_path)

    config = {
        "frozen_mode": True,
        "manifest": {
            "output_file": str(manifest_path),
            "exclusions_file": str(tmp_path / "exclusions.jsonl"),
        },
    }

    manager = CorpusManager(config)
    run1_docs = manager.load_or_retrieve()
    run2_docs = manager.load_or_retrieve()

    assert len(run1_docs) == len(run2_docs) == 5
    for d1, d2 in zip(run1_docs, run2_docs, strict=True):
        assert d1.doc_id == d2.doc_id
        assert d1.text_hash == d2.text_hash
        assert d1.model_dump_json() == d2.model_dump_json()


def test_frozen_mode_blocks_live_retrieval() -> None:
    config = {"frozen_mode": True, "query": "test"}
    manager = CorpusManager(config)
    with pytest.raises(NetworkAccessForbiddenError, match="Network access forbidden"):
        manager._retrieve_live()
