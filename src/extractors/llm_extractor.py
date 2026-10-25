"""Prompted LLM extractor implementation using Google Gemini."""

import logging
from typing import Any

from src.extractors.base import BaseExtractor
from src.extractors.llm.client import GeminiLLMClient
from src.schemas.prediction import (
    Direction,
    LanguageStatus,
    Polarity,
    Prediction,
    Provenance,
    RelationClass,
    RelationLabel,
    ReviewStatus,
)
from src.schemas.text import CandidatePair
from src.utils.config import PROJECT_ROOT, load_yaml_config

logger = logging.getLogger(__name__)


class LLMExtractor(BaseExtractor):
    """
    Extractor C: Prompted LLM using Google Gemini with structured JSON outputs and schema validation.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        client: GeminiLLMClient | None = None,
    ) -> None:
        if config is None:
            models_cfg = load_yaml_config(PROJECT_ROOT / "configs" / "models.yaml")
            self.config = models_cfg.get("extractors", {}).get("llm", {})
        else:
            self.config = config

        self.name: str = self.config.get("name", "Prompted-LLM-Gemini")
        self.version: str = self.config.get("version", "1.0.0")
        self.model_id: str = self.config.get("model_id", "gemini-3.5-flash")

        # Load prompts
        prompts_cfg = load_yaml_config(PROJECT_ROOT / "configs" / "prompts.yaml")
        self.system_prompt: str = prompts_cfg["system_prompt"]
        self.user_template: str = prompts_cfg["user_template"]

        if client is not None:
            self.client = client
        else:
            self.client = GeminiLLMClient(
                api_key=self.config.get("api_key"),
                base_url=self.config.get(
                    "base_url", "https://generativelanguage.googleapis.com/v1beta/openai"
                ),
                model_id=self.model_id,
                temperature=float(self.config.get("temperature", 0.0)),
                cache_path=self.config.get("cache_path", "data/cache/llm_cache.jsonl"),
            )

        self.parse_failures: int = 0

    def get_client_stats(self) -> dict[str, int]:
        """Return cache and request statistics from the underlying LLM client."""
        if hasattr(self.client, "get_stats"):
            return self.client.get_stats()  # type: ignore[no-any-return]
        return {"cache_hits": 0, "cache_misses": 0, "api_calls": 0, "failed_calls": 0}

    def predict_single(self, pair: CandidatePair) -> Prediction:
        """Extract relation for a single CandidatePair."""
        user_prompt = self.user_template.format(
            doc_id=pair.doc_id,
            sent_id=pair.sent_id,
            sentence_text=pair.sentence_text,
            entity_1_text=pair.entity_1.surface_text,
            e1_start=pair.entity_1.start_char,
            e1_end=pair.entity_1.end_char,
            entity_1_canonical=pair.entity_1.canonical_node_id,
            entity_2_text=pair.entity_2.surface_text,
            e2_start=pair.entity_2.start_char,
            e2_end=pair.entity_2.end_char,
            entity_2_canonical=pair.entity_2.canonical_node_id,
        )

        provenance = Provenance(
            doc_id=pair.doc_id,
            sent_id=pair.sent_id,
            url=f"https://europepmc.org/article/MED/{pair.doc_id.split(':')[-1]}",
            pmid=pair.doc_id.split(":")[-1] if pair.doc_id.startswith("PMID:") else None,
            sentence_text=pair.sentence_text,
            char_spans=[pair.entity_1.start_char, pair.entity_2.end_char],
            extractor_name=self.name,
            confidence=1.0,
            review_status=ReviewStatus.UNREVIEWED,
        )

        try:
            raw_response = self.client.generate_json(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                pair_id=pair.pair_id,
            )
        except Exception as e:
            self.parse_failures += 1
            logger.warning("LLM API query error for pair %s: %s", pair.pair_id, e)
            fallback_label = RelationLabel(
                relation_class=RelationClass.NONE,
                direction=Direction.NOT_APPLICABLE,
                polarity=Polarity.NOT_APPLICABLE,
                language_status=LanguageStatus.AFFIRMED,
                confidence=0.0,
                reasoning=f"Query error: {e}",
                provenance=provenance,
            )
            return Prediction(
                pair_id=pair.pair_id,
                extractor_name=self.name,
                extractor_version=self.version,
                label=fallback_label,
                raw_output=str(e),
                validation_status="parse_error",
                error_message=str(e),
            )

        # Validate schema
        try:
            if isinstance(raw_response, list):
                if len(raw_response) > 0 and isinstance(raw_response[0], dict):
                    raw_response = raw_response[0]
                else:
                    raw_response = {}
            elif not isinstance(raw_response, dict):
                raw_response = {}

            rel_class = RelationClass(raw_response.get("relation_class", "none"))
            direction = Direction(raw_response.get("direction", "not_applicable"))
            polarity = Polarity(raw_response.get("polarity", "not_applicable"))

            raw_lang = raw_response.get("language_status", "affirmed")
            if raw_lang in ("not_applicable", "n/a", "none", None):
                raw_lang = "affirmed"
            lang_status = LanguageStatus(raw_lang)

            conf = float(raw_response.get("confidence", 1.0))
            reasoning = str(raw_response.get("reasoning", ""))

            label = RelationLabel(
                relation_class=rel_class,
                direction=direction,
                polarity=polarity,
                language_status=lang_status,
                confidence=conf,
                reasoning=reasoning,
                provenance=provenance,
            )

            return Prediction(
                pair_id=pair.pair_id,
                extractor_name=self.name,
                extractor_version=self.version,
                label=label,
                raw_output=raw_response,
                validation_status="valid",
            )
        except Exception as e:
            self.parse_failures += 1
            logger.warning("Schema validation error for pair %s: %s", pair.pair_id, e)
            fallback_label = RelationLabel(
                relation_class=RelationClass.NONE,
                direction=Direction.NOT_APPLICABLE,
                polarity=Polarity.NOT_APPLICABLE,
                language_status=LanguageStatus.AFFIRMED,
                confidence=0.0,
                reasoning=f"Schema error: {e}",
                provenance=provenance,
            )
            return Prediction(
                pair_id=pair.pair_id,
                extractor_name=self.name,
                extractor_version=self.version,
                label=fallback_label,
                raw_output=raw_response,
                validation_status="schema_error",
                error_message=str(e),
            )

    def predict(self, pairs: list[CandidatePair]) -> list[Prediction]:
        """Run extraction on all candidate pairs."""
        return [self.predict_single(p) for p in pairs]
