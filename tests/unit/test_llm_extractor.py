"""Unit tests for Groq LLM Extractor, caching, and schema validation."""

from pathlib import Path
from unittest.mock import MagicMock

from src.extractors.llm.cache import LLMCache, compute_cache_key
from src.extractors.llm.client import GroqLLMClient
from src.extractors.llm_extractor import LLMExtractor
from src.schemas.prediction import Direction, Polarity, RelationClass
from src.schemas.text import CandidatePair, Mention


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache_file = tmp_path / "test_cache.jsonl"
    cache = LLMCache(cache_file=cache_file)

    key = compute_cache_key("llama-3.3-70b", "prompt text", "PAIR:01")
    assert cache.get(key) is None

    cache.set(key, {"relation_class": "causal", "confidence": 0.95})
    assert cache.get(key) == {"relation_class": "causal", "confidence": 0.95}

    # Re-instantiate from file
    cache2 = LLMCache(cache_file=cache_file)
    assert cache2.get(key) == {"relation_class": "causal", "confidence": 0.95}


def test_llm_extractor_predict_with_mock_client() -> None:
    mock_client = MagicMock(spec=GroqLLMClient)
    mock_client.generate_json.return_value = {
        "relation_class": "causal",
        "direction": "entity_1->entity_2",
        "polarity": "decreases",
        "language_status": "affirmed",
        "confidence": 0.98,
        "reasoning": "Text explicitly states exercise reduces risk.",
    }

    extractor = LLMExtractor(client=mock_client)

    pair = CandidatePair(
        pair_id="TEST:001",
        doc_id="PMID:111",
        sent_id="PMID:111:s01",
        sentence_text="Exercise reduces the risk of diabetes.",
        entity_1=Mention(
            mention_id="m1",
            sent_id="PMID:111:s01",
            doc_id="PMID:111",
            surface_text="Exercise",
            start_char=0,
            end_char=8,
            canonical_node_id="PhysActivity",
        ),
        entity_2=Mention(
            mention_id="m2",
            sent_id="PMID:111:s01",
            doc_id="PMID:111",
            surface_text="diabetes",
            start_char=29,
            end_char=37,
            canonical_node_id="Outcome",
        ),
    )

    preds = extractor.predict([pair])
    assert len(preds) == 1
    pred = preds[0]
    assert pred.validation_status == "valid"
    assert pred.label.relation_class == RelationClass.CAUSAL
    assert pred.label.direction == Direction.E1_TO_E2
    assert pred.label.polarity == Polarity.DECREASES
    assert pred.label.confidence == 0.98
    assert pred.label.provenance is not None
    assert pred.label.provenance.doc_id == "PMID:111"


def test_llm_extractor_schema_error_handling() -> None:
    mock_client = MagicMock(spec=GroqLLMClient)
    # Returns invalid relation class
    mock_client.generate_json.return_value = {
        "relation_class": "invalid_nonexistent_class",
        "direction": "entity_1->entity_2",
    }

    extractor = LLMExtractor(client=mock_client)

    pair = CandidatePair(
        pair_id="TEST:002",
        doc_id="PMID:222",
        sent_id="PMID:222:s01",
        sentence_text="Sample sentence.",
        entity_1=Mention(
            mention_id="m1",
            sent_id="s1",
            doc_id="PMID:222",
            surface_text="A",
            start_char=0,
            end_char=1,
            canonical_node_id="A",
        ),
        entity_2=Mention(
            mention_id="m2",
            sent_id="s1",
            doc_id="PMID:222",
            surface_text="B",
            start_char=2,
            end_char=3,
            canonical_node_id="B",
        ),
    )

    preds = extractor.predict([pair])
    assert len(preds) == 1
    assert preds[0].validation_status == "schema_error"
    assert extractor.parse_failures == 1
