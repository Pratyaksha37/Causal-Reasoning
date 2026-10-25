"""Manifest serialization, deserialization, and integrity validation."""

import json
import logging
from pathlib import Path

from src.schemas.document import Document, compute_text_hash

logger = logging.getLogger(__name__)


def write_manifest(
    documents: list[Document],
    output_path: Path | str,
) -> None:
    """
    Write document corpus manifest to a JSON Lines file.

    Args:
        documents: List of Document objects.
        output_path: Target .jsonl file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.model_dump_json() + "\n")

    logger.info("Saved %d documents to manifest: %s", len(documents), path)


def read_manifest(
    manifest_path: Path | str,
    verify_hashes: bool = True,
) -> list[Document]:
    """
    Load and validate documents from a manifest JSON Lines file.

    Args:
        manifest_path: Path to .jsonl file.
        verify_hashes: If True, re-calculate SHA-256 and assert it matches text_hash.

    Returns:
        List of validated Document objects.

    Raises:
        FileNotFoundError: If manifest file does not exist.
        ValueError: If a document fails integrity verification.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus manifest file not found: {path}")

    documents: list[Document] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            doc = Document.model_validate(data)

            if verify_hashes:
                expected_hash = compute_text_hash(doc.abstract_text)
                if doc.text_hash != expected_hash:
                    raise ValueError(
                        f"Integrity check failed for document {doc.doc_id} on line {line_num}: "
                        f"recorded hash {doc.text_hash} != calculated hash {expected_hash}"
                    )
            documents.append(doc)

    logger.info(
        "Loaded %d documents from manifest %s (hash verification=%s)",
        len(documents),
        path,
        verify_hashes,
    )
    return documents


def write_exclusions(
    exclusions: list[dict[str, str]],
    output_path: Path | str,
) -> None:
    """Save excluded records log to JSON Lines file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in exclusions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Saved %d exclusion entries to %s", len(exclusions), path)
