"""Unit tests for publication deduplication logic."""

from src.retrieval.dedup import deduplicate_documents
from src.schemas.document import Document


def test_dedup_pmid_priority() -> None:
    doc1 = Document.create(
        title="Study 1",
        abstract_text="Abstract text 1",
        pub_date="2024",
        query_provenance="query",
        retrieval_date="2026-10-22",
        pmid="1001",
        doi="10.1001/a",
    )
    doc2 = Document.create(
        title="Study 1 duplicate",
        abstract_text="Abstract text 1 duplicate",
        pub_date="2024",
        query_provenance="query",
        retrieval_date="2026-10-22",
        pmid="1001",  # Same PMID
        doi="10.1001/b",
    )
    doc3 = Document.create(
        title="Study 2",
        abstract_text="Abstract text 2",
        pub_date="2024",
        query_provenance="query",
        retrieval_date="2026-10-22",
        pmid="1002",
        doi="10.1002/a",
    )

    unique, exclusions = deduplicate_documents([doc1, doc2, doc3])
    assert len(unique) == 2
    assert unique[0].pmid == "1001"
    assert unique[1].pmid == "1002"
    assert len(exclusions) == 1
    assert "Duplicate PMID: 1001" in exclusions[0]["reason"]


def test_dedup_text_hash_duplicate() -> None:
    # Different titles / no IDs, but identical text
    doc1 = Document.create(
        title="Study A",
        abstract_text="Identical abstract content across both records.",
        pub_date="2023",
        query_provenance="query",
        retrieval_date="2026-10-22",
    )
    doc2 = Document.create(
        title="Study B",
        abstract_text="Identical abstract content across both records.",
        pub_date="2023",
        query_provenance="query",
        retrieval_date="2026-10-22",
    )

    unique, exclusions = deduplicate_documents([doc1, doc2])
    assert len(unique) == 1
    assert len(exclusions) == 1
    assert "Duplicate text hash" in exclusions[0]["reason"]
