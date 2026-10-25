"""Sentence splitting with character offset tracking."""

import spacy
from spacy.language import Language

from src.schemas.text import Sentence


class SentenceSplitter:
    """Splits document abstract text into Sentence objects with character offsets."""

    def __init__(self, nlp: Language | None = None) -> None:
        if nlp is not None:
            self.nlp = nlp
        else:
            # Create a lightweight sentencizer pipeline
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def split(self, doc_id: str, text: str) -> list[Sentence]:
        """
        Split raw text into Sentence objects.

        Args:
            doc_id: Parent document identifier.
            text: Raw text to split.

        Returns:
            List of Sentence objects with exact start_char and end_char offsets.
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        sentences: list[Sentence] = []

        for idx, span in enumerate(doc.sents):
            s_text = span.text.strip()
            if not s_text:
                continue

            # Compute stripped character offsets within the original text
            start_offset = span.start_char + (len(span.text) - len(span.text.lstrip()))
            end_offset = start_offset + len(s_text)

            sent_id = f"{doc_id}:s{idx + 1:02d}"
            sentences.append(
                Sentence(
                    doc_id=doc_id,
                    sent_id=sent_id,
                    sentence_idx=idx,
                    text=s_text,
                    start_char=start_offset,
                    end_char=end_offset,
                )
            )

        return sentences
