"""Candidate generation: mention extraction, alias mapping, and sentence-level pair cross-product."""

import itertools
import logging
import re

from src.mapping.alias_mapper import AliasMapper
from src.schemas.text import CandidatePair, Mention, Sentence

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """Extracts mentions from sentences and forms CandidatePair instances."""

    def __init__(self, alias_mapper: AliasMapper | None = None) -> None:
        self.alias_mapper = alias_mapper or AliasMapper()

    def extract_mentions(self, sentence: Sentence) -> list[Mention]:
        """
        Extract non-overlapping entity mentions from sentence using compiled alias patterns.

        Args:
            sentence: Sentence object containing text and offsets.

        Returns:
            List of non-overlapping Mention objects sorted by start character offset.
        """
        text = sentence.text
        mentions: list[Mention] = []
        occupied_spans: list[tuple[int, int]] = []

        # Find occurrences of known alias phrases
        for alias_phrase, _canonical_id, _val_state, _flip in self.alias_mapper.alias_entries:
            pattern = re.compile(rf"\b{re.escape(alias_phrase)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()

                # Check for overlap with already extracted longer spans
                if any(
                    start < span_end and end > span_start for span_start, span_end in occupied_spans
                ):
                    continue

                surface = text[start:end]
                # Map using precision rules
                mapping_info = self.alias_mapper.map_mention(surface)

                mention_id = f"{sentence.sent_id}:m{len(mentions) + 1:02d}"
                mention = Mention(
                    mention_id=mention_id,
                    sent_id=sentence.sent_id,
                    doc_id=sentence.doc_id,
                    surface_text=surface,
                    start_char=start,
                    end_char=end,
                    entity_type="CONCEPT",
                    canonical_node_id=mapping_info["canonical_node_id"],
                    alias_matched=mapping_info["alias_matched"],
                    polarity_flip=mapping_info["polarity_flip"],
                    value_state=mapping_info["value_state"],
                )
                mentions.append(mention)
                occupied_spans.append((start, end))

        # Check for vague smoking mentions that should be explicitly marked unmapped
        for pattern in self.alias_mapper.smoker_unmapped_patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if any(
                    start < span_end and end > span_start for span_start, span_end in occupied_spans
                ):
                    continue
                surface = text[start:end]
                mention_id = f"{sentence.sent_id}:m{len(mentions) + 1:02d}"
                mentions.append(
                    Mention(
                        mention_id=mention_id,
                        sent_id=sentence.sent_id,
                        doc_id=sentence.doc_id,
                        surface_text=surface,
                        start_char=start,
                        end_char=end,
                        entity_type="CONCEPT",
                        canonical_node_id="unmapped",
                        alias_matched=None,
                        polarity_flip=False,
                        value_state=None,
                    )
                )
                occupied_spans.append((start, end))

        # Sort mentions by character start position
        mentions.sort(key=lambda m: m.start_char)
        return mentions

    def generate_candidate_pairs(
        self,
        sentence: Sentence,
        mentions: list[Mention] | None = None,
        split: str = "unassigned",
    ) -> list[CandidatePair]:
        """
        Generate all ordered co-occurring mention pairs within a single sentence.

        Args:
            sentence: Sentence object.
            mentions: Pre-extracted mentions, or None to extract automatically.
            split: Document split assignment ('train', 'dev', 'test', 'unassigned').

        Returns:
            List of CandidatePair objects.
        """
        if mentions is None:
            mentions = self.extract_mentions(sentence)

        if len(mentions) < 2:
            return []

        pairs: list[CandidatePair] = []
        # Generate 2-combinations preserving text order (e1 precedes e2)
        for pair_idx, (e1, e2) in enumerate(itertools.combinations(mentions, 2), start=1):
            pair_id = f"{sentence.sent_id}:p{pair_idx:02d}"
            pairs.append(
                CandidatePair(
                    pair_id=pair_id,
                    doc_id=sentence.doc_id,
                    sent_id=sentence.sent_id,
                    sentence_text=sentence.text,
                    entity_1=e1,
                    entity_2=e2,
                    split=split,  # type: ignore[arg-type]
                )
            )

        return pairs
