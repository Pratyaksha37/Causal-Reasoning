"""Rule-based relation extractor using spaCy dependency parse, lexical triggers, and negation/hedging detection."""

import logging
import re
from typing import Any

from src.extractors.base import BaseExtractor
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
from src.text.pipeline import TextPipeline
from src.utils.config import PROJECT_ROOT, load_yaml_config

logger = logging.getLogger(__name__)


class RulesExtractor(BaseExtractor):
    """
    Extractor A: Interpretable rule-based extractor using trigger phrases and syntactic patterns.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        nlp_pipeline: TextPipeline | None = None,
    ) -> None:
        if config is None:
            models_cfg = load_yaml_config(PROJECT_ROOT / "configs" / "models.yaml")
            self.config = models_cfg.get("extractors", {}).get("rules", {})
        else:
            self.config = config

        self.name: str = self.config.get("name", "spaCy-Rules")
        self.version: str = self.config.get("version", "1.0.0")

        self.causal_triggers: list[str] = self.config.get("causal_triggers", [])
        self.association_triggers: list[str] = self.config.get("association_triggers", [])
        self.negation_triggers: list[str] = self.config.get("negation_triggers", [])
        self.hedging_triggers: list[str] = self.config.get("hedging_triggers", [])

        # Decreasing triggers for polarity
        self.decrease_triggers = [
            "reduce", "reduces", "reduced", "reducing",
            "decrease", "decreases", "decreased", "decreasing",
            "lower", "lowers", "lowered", "lowering",
            "prevent", "prevents", "prevented", "preventing",
            "protect", "protects", "protective",
        ]

        self.pipeline = nlp_pipeline or TextPipeline()

    def predict_single(self, pair: CandidatePair) -> Prediction:
        """Extract relation for a single CandidatePair using rules."""
        text = pair.sentence_text
        text_lower = text.lower()

        pair.entity_1.surface_text.lower()
        pair.entity_2.surface_text.lower()

        # Check between-spans text segment
        start_between = min(pair.entity_1.end_char, pair.entity_2.end_char)
        end_between = max(pair.entity_1.start_char, pair.entity_2.start_char)
        text[start_between:end_between].lower().strip() if start_between < end_between else ""

        # 1. Detect Negation
        is_negated = any(
            re.search(rf"\b{re.escape(neg)}\b", text_lower)
            for neg in self.negation_triggers
        )
        lang_status = LanguageStatus.NEGATED if is_negated else LanguageStatus.AFFIRMED

        # 2. Detect Hedging (if not negated)
        if not is_negated:
            is_hedged = any(
                re.search(rf"\b{re.escape(hedge)}\b", text_lower)
                for hedge in self.hedging_triggers
            )
            if is_hedged:
                lang_status = LanguageStatus.UNCERTAIN

        # 3. Detect Causal Triggers
        matched_causal: str | None = None
        for trigger in self.causal_triggers:
            # Avoid compound matches like 'all-cause', 'cause-specific', 'non-cause'
            pattern = rf"(?<!all-)(?<!cause-)(?<!non-)\b{re.escape(trigger)}\b(?!-specific)"
            if re.search(pattern, text_lower):
                matched_causal = trigger
                break

        # 4. Detect Association Triggers
        matched_assoc: str | None = None
        for trigger in self.association_triggers:
            if re.search(rf"\b{re.escape(trigger)}\b", text_lower):
                matched_assoc = trigger
                break

        provenance = Provenance(
            doc_id=pair.doc_id,
            sent_id=pair.sent_id,
            url=f"https://europepmc.org/article/MED/{pair.doc_id.split(':')[-1]}",
            pmid=pair.doc_id.split(":")[-1] if pair.doc_id.startswith("PMID:") else None,
            sentence_text=pair.sentence_text,
            char_spans=[pair.entity_1.start_char, pair.entity_2.end_char],
            extractor_name=self.name,
            confidence=0.85 if matched_causal else (0.80 if matched_assoc else 0.50),
            review_status=ReviewStatus.UNREVIEWED,
        )

        # Classification decision
        if matched_causal:
            rel_class = RelationClass.CAUSAL

            # Direction heuristic: check passive voice or entity order
            is_passive = bool(re.search(r"\b(is|are|was|were|caused by|induced by|triggered by)\b", text_lower))
            if is_passive and "caused by" in text_lower or "induced by" in text_lower or "triggered by" in text_lower:
                direction = Direction.E2_TO_E1
            elif pair.entity_1.start_char < pair.entity_2.start_char:
                direction = Direction.E1_TO_E2
            else:
                direction = Direction.E2_TO_E1

            # Polarity heuristic
            is_decrease = any(re.search(rf"\b{re.escape(dt)}\b", text_lower) for dt in self.decrease_triggers)
            polarity = Polarity.DECREASES if is_decrease else Polarity.INCREASES

            reasoning = f"Matched causal trigger '{matched_causal}' (passive={is_passive}, decrease={is_decrease})"

        elif matched_assoc:
            rel_class = RelationClass.ASSOCIATION_ONLY
            direction = Direction.NOT_APPLICABLE
            is_decrease = any(re.search(rf"\b{re.escape(dt)}\b", text_lower) for dt in self.decrease_triggers)
            polarity = Polarity.DECREASES if is_decrease else Polarity.INCREASES
            reasoning = f"Matched association trigger '{matched_assoc}'"

        else:
            rel_class = RelationClass.NONE
            direction = Direction.NOT_APPLICABLE
            polarity = Polarity.NOT_APPLICABLE
            lang_status = LanguageStatus.AFFIRMED
            reasoning = "No causal or association triggers matched"

        label = RelationLabel(
            relation_class=rel_class,
            direction=direction,
            polarity=polarity,
            language_status=lang_status,
            confidence=0.85 if matched_causal else (0.80 if matched_assoc else 0.90),
            reasoning=reasoning,
            provenance=provenance,
        )

        return Prediction(
            pair_id=pair.pair_id,
            extractor_name=self.name,
            extractor_version=self.version,
            label=label,
            raw_output={"matched_causal": matched_causal, "matched_assoc": matched_assoc},
            validation_status="valid",
        )

    def predict(self, pairs: list[CandidatePair]) -> list[Prediction]:
        """Run rule-based extraction across candidate pairs."""
        return [self.predict_single(p) for p in pairs]
