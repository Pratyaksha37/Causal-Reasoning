"""Corpus management coordinating live Europe PMC retrieval and frozen-mode loading."""

import argparse
import logging
from pathlib import Path
from typing import Any

from src.retrieval.dedup import deduplicate_documents
from src.retrieval.epmc_client import EuropePMCClient
from src.retrieval.manifest import read_manifest, write_exclusions, write_manifest
from src.schemas.document import Document
from src.utils.config import load_yaml_config
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class NetworkAccessForbiddenError(RuntimeError):
    """Raised when an attempt to access external network is made while in frozen mode."""

    pass


class CorpusManager:
    """Manages frozen corpus loading and live Europe PMC retrieval."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.frozen_mode: bool = bool(config.get("frozen_mode", False))
        self.manifest_file: str = config.get("manifest", {}).get(
            "output_file", "data/manifest/corpus_manifest.jsonl"
        )
        self.exclusions_file: str = config.get("manifest", {}).get(
            "exclusions_file", "data/manifest/exclusions.jsonl"
        )

    def load_or_retrieve(self, force_live: bool = False) -> list[Document]:
        """
        Load corpus from manifest if available or frozen; retrieve live if not frozen and manifest missing.

        Args:
            force_live: If True and frozen_mode is False, force re-retrieval from Europe PMC.

        Returns:
            List of unique, verified Document objects.
        """
        manifest_path = Path(self.manifest_file)

        if self.frozen_mode:
            logger.info("FROZEN MODE active: loading exclusively from manifest %s", manifest_path)
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Frozen mode is active but manifest file does not exist: {manifest_path}"
                )
            return read_manifest(manifest_path, verify_hashes=True)

        if manifest_path.exists() and not force_live:
            logger.info("Manifest %s exists. Loading existing corpus.", manifest_path)
            return read_manifest(manifest_path, verify_hashes=True)

        # Live retrieval path
        return self._retrieve_live()

    def _retrieve_live(self) -> list[Document]:
        """Execute live Europe PMC retrieval, deduplicate, and freeze to manifest."""
        if self.frozen_mode:
            raise NetworkAccessForbiddenError(
                "Network access forbidden: frozen_mode is enabled in configuration."
            )

        api_cfg = self.config.get("api", {})
        client = EuropePMCClient(
            base_url=api_cfg.get(
                "base_url", "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            ),
            timeout=int(api_cfg.get("timeout_seconds", 30)),
            max_retries=int(api_cfg.get("max_retries", 5)),
            backoff_factor=float(api_cfg.get("backoff_factor", 1.5)),
            rate_limit_per_second=float(api_cfg.get("rate_limit_per_second", 5.0)),
            email=api_cfg.get("email"),
        )

        query = self.config["query"]
        max_results = int(self.config.get("max_results", 500))
        page_size = int(self.config.get("page_size", 100))
        sort_by = self.config.get("sort_by", "CITED")

        raw_docs, exclusions_api = client.retrieve_corpus(
            query=query,
            max_results=max_results,
            page_size=page_size,
            sort_by=sort_by,
        )

        unique_docs, exclusions_dedup = deduplicate_documents(raw_docs)
        all_exclusions = exclusions_api + exclusions_dedup

        # Freeze results to manifest
        write_manifest(unique_docs, self.manifest_file)
        write_exclusions(all_exclusions, self.exclusions_file)

        return unique_docs


def main(config_path: str | None = None, force_live: bool = False) -> None:
    """CLI entry point for corpus retrieval / loading."""
    setup_logging()
    cfg_file = config_path or "configs/retrieval.yaml"
    config = load_yaml_config(cfg_file)
    manager = CorpusManager(config)
    docs = manager.load_or_retrieve(force_live=force_live)
    print(f"Successfully processed corpus: {len(docs)} documents loaded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Europe PMC retrieval and corpus manager")
    parser.add_argument(
        "--config", default="configs/retrieval.yaml", help="Path to retrieval config YAML"
    )
    parser.add_argument(
        "--force-live", action="store_true", help="Force live retrieval even if manifest exists"
    )
    args = parser.parse_args()
    main(config_path=args.config, force_live=args.force_live)
