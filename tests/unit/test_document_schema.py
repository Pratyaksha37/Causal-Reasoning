"""Unit tests for Document schema, ID resolution, and SHA-256 hashing."""

import pytest
from pydantic import ValidationError

from src.schemas.document import Document, compute_text_hash


def test_compute_text_hash_deterministic() -> None:
    text1 = "Physical activity decreases the risk of type 2 diabetes mellitus."
    text2 = "Physical   activity decreases the risk of type 2 diabetes mellitus. "
    # Normalized whitespace should produce the exact same hash
    assert compute_text_hash(text1) == compute_text_hash(text2)
    assert len(compute_text_hash(text1)) == 64


def test_document_create_pmid_priority() -> None:
    doc = Document.create(
        title="Physical Activity and Diabetes",
        abstract_text="Exercise prevents diabetes.",
        pub_date="2024",
        query_provenance="test query",
        retrieval_date="2026-10-22",
        pmid="12345678",
        pmcid="PMC9876543",
        doi="10.1000/182",
    )
    assert doc.doc_id == "PMID:12345678"
    assert doc.pmid == "12345678"
    assert doc.text_hash == compute_text_hash("Exercise prevents diabetes.")


def test_document_create_fallback_ids() -> None:
    doc_pmc = Document.create(
        title="Title",
        abstract_text="Abstract text",
        pub_date="2024",
        query_provenance="test query",
        retrieval_date="2026-10-22",
        pmcid="PMC9876543",
    )
    assert doc_pmc.doc_id == "PMCID:PMC9876543"

    doc_doi = Document.create(
        title="Title",
        abstract_text="Abstract text",
        pub_date="2024",
        query_provenance="test query",
        retrieval_date="2026-10-22",
        doi="10.1234/sample",
    )
    assert doc_doi.doc_id == "DOI:10.1234/sample"


def test_document_invalid_hash_raises() -> None:
    with pytest.raises(ValidationError):
        Document(
            doc_id="PMID:1",
            title="Title",
            pub_date="2024",
            abstract_text="Text",
            text_hash="short_hash",  # Invalid hash length
            query_provenance="query",
            retrieval_date="2026-10-22",
        )
