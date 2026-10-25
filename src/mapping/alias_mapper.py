"""Alias-to-canonical variable mapping with precision rules and polarity flipping."""

import re
from typing import Any

from src.mapping.registry import CanonicalRegistry


class AliasMapper:
    """
    Maps surface text entity mentions to canonical variables or open-domain concepts.
    Enforces strict CDC criteria, polarity-flipping on inactivity, and unmapped fallbacks.
    """

    def __init__(self, registry: CanonicalRegistry | None = None) -> None:
        self.registry = registry or CanonicalRegistry()
        self._compile_lookup_tables()

    def _compile_lookup_tables(self) -> None:
        """Compile regexes and phrase lists for fast, accurate matching."""
        # 1. Smoker strict patterns
        self.smoker_1_patterns = [
            re.compile(
                r"\b(ever\s+smoked\s+at\s+least\s+100\s+cigarettes|smoked\s*>=\s*100\s+cigarettes|100\+?\s*lifetime\s*cigarettes)\b",
                re.IGNORECASE,
            )
        ]
        self.smoker_0_patterns = [
            re.compile(
                r"\b(never\s+smoked|never\s+smoker|smoked\s+less\s+than\s+100\s+cigarettes)\b",
                re.IGNORECASE,
            )
        ]
        self.smoker_unmapped_patterns = [
            re.compile(
                r"\b(smoking|tobacco|cigarette|smoker|smoking\s+history|smoking\s+cessation|current\s+smoker)\b",
                re.IGNORECASE,
            )
        ]

        # 2. Compile dictionary of exact and normalized phrases
        # Format: (normalized_phrase, canonical_id, value_state, polarity_flip)
        self.alias_entries: list[tuple[str, str, str | None, bool]] = []

        # Canonical variables
        for var_id, var_meta in self.registry.canonical_vars.items():
            aliases_dict = var_meta.get("aliases", {})
            polarity_flip_vals = var_meta.get("polarity_flip_on_aliases", [])

            for val_state, phrases in aliases_dict.items():
                flip = str(val_state) in [str(x) for x in polarity_flip_vals]
                for phrase in phrases:
                    self.alias_entries.append(
                        (phrase.lower().strip(), var_id, str(val_state), flip)
                    )

        # Open-domain concepts
        for concept_id, concept_meta in self.registry.open_domain_concepts.items():
            for phrase in concept_meta.get("aliases", []):
                self.alias_entries.append((phrase.lower().strip(), concept_id, None, False))

        # Sort alias entries longest first so more specific phrases match before substrings
        self.alias_entries.sort(key=lambda x: len(x[0]), reverse=True)

    def map_mention(self, surface_text: str) -> dict[str, Any]:
        """
        Map a mention surface text to its canonical concept or 'unmapped'.

        Args:
            surface_text: The verbatim phrase extracted from text.

        Returns:
            Dict containing:
                - canonical_node_id: str (e.g. 'BMICat', 'PhysActivity', or 'unmapped')
                - value_state: Optional[str]
                - polarity_flip: bool
                - alias_matched: Optional[str]
        """
        text = surface_text.lower().strip()
        normalized_text = " ".join(text.split())

        # Rule 1: Strict Smoker precision check
        # If any vague smoking word is present, only map if explicit 100-cigarette criterion is met
        for pattern in self.smoker_1_patterns:
            if pattern.search(normalized_text):
                return {
                    "canonical_node_id": "Smoker",
                    "value_state": "1",
                    "polarity_flip": False,
                    "alias_matched": "ever smoked at least 100 cigarettes",
                }

        for pattern in self.smoker_0_patterns:
            if pattern.search(normalized_text):
                return {
                    "canonical_node_id": "Smoker",
                    "value_state": "0",
                    "polarity_flip": False,
                    "alias_matched": "never smoked",
                }

        for pattern in self.smoker_unmapped_patterns:
            if pattern.search(normalized_text):
                # General smoking mention that fails 100-cigarette criterion must NOT be mapped to Smoker
                return {
                    "canonical_node_id": "unmapped",
                    "value_state": None,
                    "polarity_flip": False,
                    "alias_matched": None,
                }

        # Rule 2: Search compiled alias entries
        for alias_phrase, canonical_id, val_state, flip in self.alias_entries:
            # Word boundary regex match or exact match
            escaped_alias = re.escape(alias_phrase)
            pattern = re.compile(rf"\b{escaped_alias}\b", re.IGNORECASE)

            if pattern.search(normalized_text) or alias_phrase == normalized_text:
                return {
                    "canonical_node_id": canonical_id,
                    "value_state": val_state,
                    "polarity_flip": flip,
                    "alias_matched": alias_phrase,
                }

        # Rule 3: Default unmapped fallback. Never force a match.
        return {
            "canonical_node_id": "unmapped",
            "value_state": None,
            "polarity_flip": False,
            "alias_matched": None,
        }
