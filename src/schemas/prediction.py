"""Data contracts for relation labels, extractors predictions, and evidence provenance."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class RelationClass(StrEnum):
    CAUSAL = "causal"
    ASSOCIATION_ONLY = "association_only"
    NONE = "none"


class Direction(StrEnum):
    E1_TO_E2 = "entity_1->entity_2"
    E2_TO_E1 = "entity_2->entity_1"
    NOT_APPLICABLE = "not_applicable"


class Polarity(StrEnum):
    INCREASES = "increases"
    DECREASES = "decreases"
    MIXED_OR_UNCLEAR = "mixed_or_unclear"
    NOT_APPLICABLE = "not_applicable"


class LanguageStatus(StrEnum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNCERTAIN = "uncertain"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Provenance(BaseModel):
    """Provenance tracking for a relation claim extracted from text."""

    doc_id: str = Field(..., description="Source document identifier")
    sent_id: str = Field(..., description="Sentence identifier")
    url: str | None = Field(default=None, description="External URL or Europe PMC link")
    pmid: str | None = Field(default=None, description="PubMed ID if available")
    sentence_text: str | None = Field(
        default=None, description="Verbatim text of the supporting sentence"
    )
    char_spans: list[int] | None = Field(
        default=None, description="[start, end] char span of trigger or sentence"
    )
    extractor_name: str | None = Field(
        default=None, description="Extractor that produced this evidence"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Model or annotation confidence"
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.UNREVIEWED, description="Human review status"
    )


class RelationLabel(BaseModel):
    """Typed relation annotation or prediction for a CandidatePair."""

    relation_class: RelationClass = Field(..., description="causal | association_only | none")
    direction: Direction = Field(
        ..., description="entity_1->entity_2 | entity_2->entity_1 | not_applicable"
    )
    polarity: Polarity = Field(
        ..., description="increases | decreases | mixed_or_unclear | not_applicable"
    )
    language_status: LanguageStatus = Field(..., description="affirmed | negated | uncertain")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    reasoning: str | None = Field(default=None, description="Explanation or matched trigger phrase")
    provenance: Provenance | None = Field(default=None, description="Detailed text provenance")


class Prediction(BaseModel):
    """Encapsulation of a single extractor's output on a CandidatePair."""

    pair_id: str = Field(..., description="Candidate pair ID")
    extractor_name: str = Field(..., description="Name of the extractor")
    extractor_version: str = Field(..., description="Version of the extractor")
    label: RelationLabel = Field(..., description="Extracted relation label")
    raw_output: Any | None = Field(
        default=None, description="Raw unparsed model output for auditability"
    )
    validation_status: Literal["valid", "schema_error", "parse_error"] = Field(
        default="valid",
        description="Whether the output passed strict schema validation",
    )
    error_message: str | None = Field(
        default=None, description="Error details if validation failed"
    )
