"""
Keyword Heuristic relation extractor baseline using entity marker tokens.

NOTE ON MODEL PROVENANCE:
This module implements a lightweight keyword-based heuristic baseline (Extractor B)
that operates on text tokenized with entity markers [E1]...[/E1] and [E2]...[/E2].
It does NOT load or evaluate fine-tuned neural Transformer weights.
A true fine-tuned DistilBERT / Bio-Transformer relation extractor remains future work.
"""

import logging
import re
from typing import Any

import torch

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
from src.utils.config import PROJECT_ROOT, load_yaml_config

logger = logging.getLogger(__name__)


def insert_entity_markers(
    sentence_text: str,
    e1_start: int,
    e1_end: int,
    e2_start: int,
    e2_end: int,
) -> str:
    """Insert [E1], [/E1], [E2], [/E2] marker tokens into sentence text."""
    if e1_start < e2_start:
        marked = (
            sentence_text[:e1_start]
            + "[E1] "
            + sentence_text[e1_start:e1_end]
            + " [/E1]"
            + sentence_text[e1_end:e2_start]
            + "[E2] "
            + sentence_text[e2_start:e2_end]
            + " [/E2]"
            + sentence_text[e2_end:]
        )
    else:
        marked = (
            sentence_text[:e2_start]
            + "[E2] "
            + sentence_text[e2_start:e2_end]
            + " [/E2]"
            + sentence_text[e2_end:e1_start]
            + "[E1] "
            + sentence_text[e1_start:e1_end]
            + " [/E1]"
            + sentence_text[e1_end:]
        )
    return " ".join(marked.split())


class KeywordHeuristicExtractor(BaseExtractor):
    """
    Extractor B: Baseline relation extractor using keyword matching with entity marker tokens.
    Note: Operates as a fast heuristic baseline; does not run trained neural weights.
    A genuine fine-tuned Transformer model remains future work.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        model_path: str | None = None,
    ) -> None:
        if config is None:
            models_cfg = load_yaml_config(PROJECT_ROOT / "configs" / "models.yaml")
            self.config = models_cfg.get("extractors", {}).get("distilbert", {})
        else:
            self.config = config

        self.name: str = self.config.get("name", "KeywordHeuristic")
        self.version: str = self.config.get("version", "1.0.0")
        self.base_model_name: str = self.config.get("base_model", "distilbert-base-uncased")
        self.max_length: int = int(self.config.get("max_length", 128))

        self.classes = [RelationClass.CAUSAL, RelationClass.ASSOCIATION_ONLY, RelationClass.NONE]

        # Check PyTorch device (MPS for Apple Silicon, CUDA, or CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        logger.info("KeywordHeuristicExtractor initialized (heuristic baseline, device: %s)", self.device)

    def predict_single(self, pair: CandidatePair) -> Prediction:
        """Extract relation for single CandidatePair with entity marker tokenization."""
        marked_text = insert_entity_markers(
            pair.sentence_text,
            pair.entity_1.start_char,
            pair.entity_1.end_char,
            pair.entity_2.start_char,
            pair.entity_2.end_char,
        )

        text_lower = pair.sentence_text.lower()
        causal_patterns = [
            r"(?<!all-)(?<!cause-)(?<!non-)\bcauses?\b(?!-specific)",
            r"\bleads?\s+to\b",
            r"\binduces?\b",
            r"\breduces?\b",
            r"\bincreases?\b",
            r"\bprevents?\b",
            r"\bcausal\s+links?\b",
        ]
        assoc_patterns = [
            r"\bassociated\b",
            r"\bcorrelated\b",
            r"\blinked\b",
            r"\brelated\b",
            r"\bpredicts?\b",
        ]

        is_causal = any(re.search(pat, text_lower) for pat in causal_patterns)
        is_assoc = any(re.search(pat, text_lower) for pat in assoc_patterns)

        if is_causal:
            rel_class = RelationClass.CAUSAL
            direction = (
                Direction.E2_TO_E1
                if "caused by" in text_lower or "induced by" in text_lower
                else Direction.E1_TO_E2
            )
            polarity = (
                Polarity.DECREASES
                if any(w in text_lower for w in ["reduce", "reduces", "prevent", "decrease", "protect"])
                else Polarity.INCREASES
            )
            confidence = 0.88
            reasoning = "Keyword heuristic classification: causal"
        elif is_assoc:
            rel_class = RelationClass.ASSOCIATION_ONLY
            direction = Direction.NOT_APPLICABLE
            polarity = Polarity.INCREASES
            confidence = 0.82
            reasoning = "Keyword heuristic classification: association_only"
        else:
            rel_class = RelationClass.NONE
            direction = Direction.NOT_APPLICABLE
            polarity = Polarity.NOT_APPLICABLE
            confidence = 0.92
            reasoning = "Keyword heuristic classification: none"

        lang_status = LanguageStatus.NEGATED if "not " in text_lower or "no " in text_lower else (
            LanguageStatus.UNCERTAIN if "may " in text_lower or "might " in text_lower or "could " in text_lower else LanguageStatus.AFFIRMED
        )

        provenance = Provenance(
            doc_id=pair.doc_id,
            sent_id=pair.sent_id,
            url=f"https://europepmc.org/article/MED/{pair.doc_id.split(':')[-1]}",
            pmid=pair.doc_id.split(":")[-1] if pair.doc_id.startswith("PMID:") else None,
            sentence_text=pair.sentence_text,
            char_spans=[pair.entity_1.start_char, pair.entity_2.end_char],
            extractor_name=self.name,
            confidence=confidence,
            review_status=ReviewStatus.UNREVIEWED,
        )

        label = RelationLabel(
            relation_class=rel_class,
            direction=direction,
            polarity=polarity,
            language_status=lang_status,
            confidence=confidence,
            reasoning=reasoning,
            provenance=provenance,
        )

        return Prediction(
            pair_id=pair.pair_id,
            extractor_name=self.name,
            extractor_version=self.version,
            label=label,
            raw_output={"marked_text": marked_text},
            validation_status="valid",
        )

    def predict(self, pairs: list[CandidatePair]) -> list[Prediction]:
        """Run extraction on candidate pairs."""
        return [self.predict_single(p) for p in pairs]


# Alias for backward compatibility
DistilBERTExtractor = KeywordHeuristicExtractor
