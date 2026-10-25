"""Biomedical and general text processing pipeline wrapping spaCy."""

import logging

import spacy
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)


class TextPipeline:
    """Wrapper around spaCy NLP pipeline with metadata logging."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        disable_pipes: list[str] | None = None,
    ) -> None:
        self.model_name = model_name
        disable = disable_pipes or []
        try:
            self.nlp: Language = spacy.load(model_name, disable=disable)
            self.model_version = self.nlp.meta.get("version", "unknown")
            logger.info("Loaded spaCy model '%s' (version: %s)", model_name, self.model_version)
        except OSError:
            logger.warning(
                "spaCy model '%s' not found. Falling back to blank 'en' pipeline.", model_name
            )
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")
            self.model_version = "blank"

    def process(self, text: str) -> Doc:
        """Process text through the spaCy pipeline."""
        return self.nlp(text)
