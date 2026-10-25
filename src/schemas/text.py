"""Data contracts for biomedical text processing, entity mentions, and candidate pairs."""

from typing import Literal

from pydantic import BaseModel, Field


class Sentence(BaseModel):
    """A sentence extracted from a publication abstract."""

    doc_id: str = Field(..., description="Parent document identifier")
    sent_id: str = Field(..., description="Unique sentence identifier (e.g., PMID:123:s01)")
    sentence_idx: int = Field(..., description="0-based sentence index within document")
    text: str = Field(..., description="Sentence text content")
    start_char: int = Field(..., description="Character offset start within document abstract")
    end_char: int = Field(..., description="Character offset end within document abstract")


class Mention(BaseModel):
    """An entity mention identified in a sentence."""

    mention_id: str = Field(..., description="Unique mention ID within sentence")
    sent_id: str = Field(..., description="Parent sentence identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    surface_text: str = Field(..., description="Exact string appearance in text")
    start_char: int = Field(..., description="Start character offset within sentence")
    end_char: int = Field(..., description="End character offset within sentence")
    entity_type: str = Field(default="CONCEPT", description="NER entity type")
    canonical_node_id: str = Field(
        default="unmapped",
        description="Canonical variable identifier (e.g. BMICat, PhysActivity, Outcome, or 'unmapped')",
    )
    alias_matched: str | None = Field(default=None, description="Matched phrase alias if mapped")
    polarity_flip: bool = Field(
        default=False,
        description="True if mention semantics require edge polarity flipping upon normalization (e.g. physical inactivity)",
    )
    value_state: str | None = Field(
        default=None,
        description="Assigned discrete value state if applicable (e.g. '0', '1', 'obese')",
    )


class CandidatePair(BaseModel):
    """
    Candidate pair of co-occurring entity mentions within the same sentence.
    All extractors are evaluated on identical CandidatePair instances.
    """

    pair_id: str = Field(..., description="Unique candidate pair identifier")
    doc_id: str = Field(..., description="Parent document identifier")
    sent_id: str = Field(..., description="Parent sentence identifier")
    sentence_text: str = Field(..., description="Full text of the sentence containing the pair")
    entity_1: Mention = Field(..., description="First entity mention")
    entity_2: Mention = Field(..., description="Second entity mention")
    split: Literal["train", "dev", "test", "unassigned"] = Field(
        default="unassigned",
        description="Document-level split partition",
    )
