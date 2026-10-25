"""Unit tests for manifest serialization, loading, and integrity validation."""

import json
from pathlib import Path

import pytest

from src.retrieval.manifest import read_manifest, write_exclusions, write_manifest
from src.schemas.document import Document


def test_manifest_roundtrip(tmp_path: Path) -> None:
    doc1 = Document.create(
        title="Physical Activity Study",
        abstract_text="Physical activity reduces the risk of diabetes.",
        pub_date="2023",
        query_provenance="test_query",
        retrieval_date="2026-10-22",
        pmid="20000001",
    )
    doc2 = Document.create(
        title="Obesity Study",
        abstract_text="Obesity increases the risk of T2DM.",
        pub_date="2024",
        query_provenance="test_query",
        retrieval_date="2026-10-22",
        pmid="20000002",
    )

    manifest_file = tmp_path / "manifest.jsonl"
    write_manifest([doc1, doc2], manifest_file)

    loaded_docs = read_manifest(manifest_file, verify_hashes=True)
    assert len(loaded_docs) == 2
    assert loaded_docs[0].doc_id == "PMID:20000001"
    assert loaded_docs[1].doc_id == "PMID:20000002"
    assert loaded_docs[0].abstract_text == doc1.abstract_text


def test_manifest_tampered_hash_raises(tmp_path: Path) -> None:
    doc = Document.create(
        title="Original Title",
        abstract_text="Original abstract text.",
        pub_date="2024",
        query_provenance="test_query",
        retrieval_date="2026-10-22",
        pmid="30000001",
    )
    manifest_file = tmp_path / "tampered.jsonl"
    data = doc.model_dump()
    data["abstract_text"] = "Tampered abstract text after manifest generation."
    manifest_file.write_text(json.dumps(data) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Integrity check failed"):
        read_manifest(manifest_file, verify_hashes=True)


def test_write_exclusions(tmp_path: Path) -> None:
    excl_file = tmp_path / "exclusions.jsonl"
    exclusions = [
        {"doc_id": "PMID:1", "title": "Title 1", "reason": "No abstract"},
        {"doc_id": "PMID:2", "title": "Title 2", "reason": "Duplicate PMID"},
    ]
    write_exclusions(exclusions, excl_file)
    assert excl_file.exists()
    lines = excl_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
