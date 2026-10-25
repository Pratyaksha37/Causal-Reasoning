"""Data contracts for retrieved documents and corpus manifests."""

import hashlib

from pydantic import BaseModel, Field, field_validator


def compute_text_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Document(BaseModel):
    """
    Representation of a publication retrieved from Europe PMC.

    Attributes:
        doc_id: Primary identifier (prefers PMID, then PMCID, then DOI).
        pmid: Optional PubMed ID.
        pmcid: Optional PubMed Central ID.
        doi: Optional Digital Object Identifier.
        title: Article title.
        pub_date: Publication date (YYYY or YYYY-MM-DD format).
        abstract_text: Text of abstract.
        text_hash: SHA-256 hash of abstract text.
        licence_note: Licence or copyright statement (e.g. 'cc by', 'open access', 'abstract-only').
        query_provenance: Query string used to retrieve document.
        retrieval_date: ISO 8601 date when document was retrieved.
        cited_by_count: Citation count recorded at retrieval time.
        full_text_stored: Whether full text is licensed to be stored.
    """

    doc_id: str = Field(..., description="Unique document ID (e.g., PMID:12345678)")
    pmid: str | None = Field(default=None, description="PubMed ID")
    pmcid: str | None = Field(default=None, description="PubMed Central ID")
    doi: str | None = Field(default=None, description="DOI")
    title: str = Field(..., description="Article title")
    pub_date: str = Field(..., description="Publication year or date")
    abstract_text: str = Field(..., description="Abstract text")
    text_hash: str = Field(..., description="SHA-256 text hash")
    licence_note: str = Field(default="abstract-only", description="Licence description")
    query_provenance: str = Field(..., description="Search query provenance")
    retrieval_date: str = Field(..., description="Date of retrieval")
    cited_by_count: int = Field(default=0, description="Citation count")
    full_text_stored: bool = Field(default=False, description="Full-text storage flag")

    @classmethod
    def create(
        cls,
        title: str,
        abstract_text: str,
        pub_date: str,
        query_provenance: str,
        retrieval_date: str,
        pmid: str | None = None,
        pmcid: str | None = None,
        doi: str | None = None,
        licence_note: str = "abstract-only",
        cited_by_count: int = 0,
        full_text_stored: bool = False,
    ) -> "Document":
        """Factory method that calculates deterministic doc_id and text_hash."""
        # Prioritize PMID > PMCID > DOI > hash
        if pmid:
            doc_id = f"PMID:{pmid}"
        elif pmcid:
            doc_id = f"PMCID:{pmcid}"
        elif doi:
            doc_id = f"DOI:{doi}"
        else:
            doc_id = f"HASH:{compute_text_hash(title + abstract_text)[:16]}"

        text_hash = compute_text_hash(abstract_text)
        return cls(
            doc_id=doc_id,
            pmid=pmid,
            pmcid=pmcid,
            doi=doi,
            title=title,
            pub_date=pub_date,
            abstract_text=abstract_text,
            text_hash=text_hash,
            licence_note=licence_note,
            query_provenance=query_provenance,
            retrieval_date=retrieval_date,
            cited_by_count=cited_by_count,
            full_text_stored=full_text_stored,
        )

    @field_validator("text_hash")
    @classmethod
    def validate_hash_length(cls, v: str) -> str:
        if len(v) != 64:
            raise ValueError(f"text_hash must be a 64-character SHA-256 hex string, got {len(v)}")
        return v
