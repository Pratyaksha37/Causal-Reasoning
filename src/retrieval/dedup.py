"""Deduplication logic for publications by prioritized persistent identifier."""

import logging

from src.schemas.document import Document

logger = logging.getLogger(__name__)


def deduplicate_documents(
    docs: list[Document],
) -> tuple[list[Document], list[dict[str, str]]]:
    """
    Deduplicate documents prioritizing PMID > PMCID > DOI > text_hash.

    Args:
        docs: List of Document objects to deduplicate.

    Returns:
        Tuple of (deduplicated_documents, exclusions_log).
    """
    seen_pmids: set[str] = set()
    seen_pmcids: set[str] = set()
    seen_dois: set[str] = set()
    seen_hashes: set[str] = set()

    unique_docs: list[Document] = []
    exclusions: list[dict[str, str]] = []

    for doc in docs:
        reason = ""
        if doc.pmid and doc.pmid in seen_pmids:
            reason = f"Duplicate PMID: {doc.pmid}"
        elif doc.pmcid and doc.pmcid in seen_pmcids:
            reason = f"Duplicate PMCID: {doc.pmcid}"
        elif doc.doi and doc.doi in seen_dois:
            reason = f"Duplicate DOI: {doc.doi}"
        elif doc.text_hash in seen_hashes:
            reason = f"Duplicate text hash: {doc.text_hash[:12]}"

        if reason:
            exclusions.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "reason": reason,
                }
            )
            continue

        if doc.pmid:
            seen_pmids.add(doc.pmid)
        if doc.pmcid:
            seen_pmcids.add(doc.pmcid)
        if doc.doi:
            seen_dois.add(doc.doi)
        seen_hashes.add(doc.text_hash)

        unique_docs.append(doc)

    logger.info(
        "Deduplication complete: retained %d unique docs from %d total (dropped %d duplicates)",
        len(unique_docs),
        len(docs),
        len(exclusions),
    )
    return unique_docs, exclusions
