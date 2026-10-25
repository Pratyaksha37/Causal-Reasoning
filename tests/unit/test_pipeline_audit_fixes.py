"""Validation and regression tests for Phase 1 audit fixes."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import networkx as nx
import pytest

from src.extractors.distilbert_extractor import DistilBERTExtractor, KeywordHeuristicExtractor
from src.extractors.llm.client import GeminiLLMClient
from src.extractors.llm.cache import compute_cache_key
from src.graph.builder import EvidenceGraphBuilder
from src.graph.exporter import export_graph_for_phase2
from src.graph.visualizer import plot_evidence_graph
from run_pipeline import build_graph_from_saved_predictions
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
from src.schemas.text import CandidatePair, Mention


def _create_test_pair(
    pair_id: str,
    doc_id: str,
    sent_id: str,
    e1_node: str,
    e2_node: str,
    text: str = "Test sentence.",
    e1_flip: bool = False,
    e2_flip: bool = False,
) -> CandidatePair:
    """Helper to create a typed CandidatePair for tests."""
    return CandidatePair(
        pair_id=pair_id,
        doc_id=doc_id,
        sent_id=sent_id,
        sentence_text=text,
        entity_1=Mention(
            mention_id=f"{sent_id}:m01",
            sent_id=sent_id,
            doc_id=doc_id,
            surface_text=e1_node,
            start_char=0,
            end_char=len(e1_node),
            entity_type="CONCEPT",
            canonical_node_id=e1_node,
            alias_matched=e1_node,
            polarity_flip=e1_flip,
        ),
        entity_2=Mention(
            mention_id=f"{sent_id}:m02",
            sent_id=sent_id,
            doc_id=doc_id,
            surface_text=e2_node,
            start_char=len(e1_node) + 1,
            end_char=len(e1_node) + 1 + len(e2_node),
            entity_type="CONCEPT",
            canonical_node_id=e2_node,
            alias_matched=e2_node,
            polarity_flip=e2_flip,
        ),
        split="test",
    )


def _create_test_prediction(
    pair: CandidatePair,
    extractor_name: str,
    rel_class: RelationClass = RelationClass.CAUSAL,
    direction: Direction = Direction.E1_TO_E2,
    confidence: float = 0.9,
    pmid: str | None = None,
) -> Prediction:
    """Helper to create a typed Prediction."""
    prov = Provenance(
        doc_id=pair.doc_id,
        sent_id=pair.sent_id,
        pmid=pmid,
        url=f"https://europepmc.org/article/MED/{pmid}" if pmid else None,
        sentence_text=pair.sentence_text,
        char_spans=[0, 10],
        extractor_name=extractor_name,
        confidence=confidence,
        review_status=ReviewStatus.UNREVIEWED,
    )
    label = RelationLabel(
        relation_class=rel_class,
        direction=direction,
        polarity=Polarity.INCREASES,
        language_status=LanguageStatus.AFFIRMED,
        confidence=confidence,
        reasoning="Test reasoning",
        provenance=prov,
    )
    return Prediction(
        pair_id=pair.pair_id,
        extractor_name=extractor_name,
        extractor_version="1.0.0",
        label=label,
        raw_output={},
        validation_status="valid",
    )


def test_synthetic_eval_examples_excluded_by_default(tmp_path: Path) -> None:
    """Test 1: Synthetic evaluation examples do not appear in evidence graph by default."""
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()

    corpus_file = tmp_path / "corpus.jsonl"
    synth_file = tmp_path / "synth.jsonl"

    # 1 corpus pair: BMICat -> Outcome
    c_pair = _create_test_pair("PMID:100:s01:p01", "PMID:100", "PMID:100:s01", "BMICat", "Outcome")
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write(c_pair.model_dump_json() + "\n")

    # 1 synthetic pair: SleepDuration -> InsulinResistance
    s_pair = _create_test_pair("SYN:008", "DOC:SYN04", "DOC:SYN04:s02", "SleepDuration", "InsulinResistance")
    with open(synth_file, "w", encoding="utf-8") as f:
        f.write(s_pair.model_dump_json() + "\n")

    # Prediction file containing both
    pred_c = _create_test_prediction(c_pair, "spaCy-Rules", pmid="100")
    pred_s = _create_test_prediction(s_pair, "spaCy-Rules", pmid=None)

    pred_file = pred_dir / "spacy-rules_predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write(pred_c.model_dump_json() + "\n")
        f.write(pred_s.model_dump_json() + "\n")

    # Default run: include_synthetic=False
    res_default = build_graph_from_saved_predictions(
        predictions_dir=pred_dir,
        corpus_candidates_file=corpus_file,
        synthetic_benchmark_file=synth_file,
        include_synthetic=False,
        output_dir=tmp_path / "graph_default",
        figures_dir=tmp_path / "fig_default",
    )
    g_default: nx.DiGraph = res_default["repaired_graph"]
    assert g_default.has_edge("BMICat", "Outcome")
    assert not g_default.has_node("SleepDuration")
    assert not g_default.has_node("InsulinResistance")

    # Debugging run: include_synthetic=True
    res_synth = build_graph_from_saved_predictions(
        predictions_dir=pred_dir,
        corpus_candidates_file=corpus_file,
        synthetic_benchmark_file=synth_file,
        include_synthetic=True,
        output_dir=tmp_path / "graph_synth",
        figures_dir=tmp_path / "fig_synth",
    )
    g_synth: nx.DiGraph = res_synth["repaired_graph"]
    assert g_synth.has_edge("SleepDuration", "InsulinResistance")


def test_repeated_predictions_single_evidence_count_and_agreement() -> None:
    """Test 2 & 3: Repeated predictions from three extractors on one sentence yield support_count=1 and record agreement."""
    builder = EvidenceGraphBuilder()
    pair = _create_test_pair("PMID:200:s01:p01", "PMID:200", "PMID:200:s01", "BMICat", "Outcome")

    p1 = _create_test_prediction(pair, "spaCy-Rules", confidence=0.85, pmid="200")
    p2 = _create_test_prediction(pair, "KeywordHeuristic", confidence=0.88, pmid="200")
    p3 = _create_test_prediction(pair, "Prompted-LLM-Gemini", confidence=0.95, pmid="200")

    builder.add_prediction(p1, pair)
    builder.add_prediction(p2, pair)
    builder.add_prediction(p3, pair)

    g = builder.causal_graph
    edge_data = g["BMICat"]["Outcome"]

    # Support count must be 1 unique sentence, not 3 predictions
    assert edge_data["support_count"] == 1
    assert edge_data["unique_evidence_sentences"] == 1
    assert edge_data["unique_papers_count"] == 1

    # Extractor agreement must record all 3 extractors
    assert set(edge_data["extractors"]) == {"spaCy-Rules", "KeywordHeuristic", "Prompted-LLM-Gemini"}
    assert len(edge_data["evidence"]) == 1

    ev_item = edge_data["evidence"][0]
    assert set(ev_item["supporting_extractors"]) == {"spaCy-Rules", "KeywordHeuristic", "Prompted-LLM-Gemini"}
    assert len(ev_item["predictions"]) == 3
    # Mean confidence
    assert ev_item["confidence"] == round((0.85 + 0.88 + 0.95) / 3, 4)


def test_different_evidence_sentences_increase_count() -> None:
    """Test 4: Different evidence sentences supporting the same edge increase unique evidence count."""
    builder = EvidenceGraphBuilder()
    pair1 = _create_test_pair("PMID:300:s01:p01", "PMID:300", "PMID:300:s01", "BMICat", "Outcome")
    pair2 = _create_test_pair("PMID:300:s02:p01", "PMID:300", "PMID:300:s02", "BMICat", "Outcome")

    builder.add_prediction(_create_test_prediction(pair1, "spaCy-Rules", pmid="300"), pair1)
    builder.add_prediction(_create_test_prediction(pair2, "spaCy-Rules", pmid="300"), pair2)

    edge_data = builder.causal_graph["BMICat"]["Outcome"]
    assert edge_data["support_count"] == 2
    assert edge_data["unique_evidence_sentences"] == 2
    assert edge_data["unique_papers_count"] == 1  # Same paper (PMID:300)


def test_different_papers_counted_separately() -> None:
    """Test 5: Different papers are counted separately when PMIDs exist."""
    builder = EvidenceGraphBuilder()
    pair1 = _create_test_pair("PMID:401:s01:p01", "PMID:401", "PMID:401:s01", "BMICat", "Outcome")
    pair2 = _create_test_pair("PMID:402:s01:p01", "PMID:402", "PMID:402:s01", "BMICat", "Outcome")

    builder.add_prediction(_create_test_prediction(pair1, "spaCy-Rules", pmid="401"), pair1)
    builder.add_prediction(_create_test_prediction(pair2, "spaCy-Rules", pmid="402"), pair2)

    edge_data = builder.causal_graph["BMICat"]["Outcome"]
    assert edge_data["support_count"] == 2
    assert edge_data["unique_papers_count"] == 2


def test_offline_graph_building_makes_zero_api_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test 6: Offline graph-building succeeds using saved predictions and makes zero API calls."""
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()

    pair = _create_test_pair("PMID:500:s01:p01", "PMID:500", "PMID:500:s01", "PhysActivity", "Outcome")
    corpus_file = tmp_path / "corpus.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        f.write(pair.model_dump_json() + "\n")

    pred = _create_test_prediction(pair, "spaCy-Rules", pmid="500")
    pred_file = pred_dir / "spacy-rules_predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        f.write(pred.model_dump_json() + "\n")

    # Mock requests.Session.post to raise if any network call is attempted
    def forbidden_post(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Network call attempted during offline graph-building!")

    monkeypatch.setattr("requests.Session.post", forbidden_post)

    result = build_graph_from_saved_predictions(
        predictions_dir=pred_dir,
        corpus_candidates_file=corpus_file,
        output_dir=tmp_path / "out_graph",
        figures_dir=tmp_path / "out_fig",
    )
    assert result["predictions_loaded"] == 1
    assert result["post_repair_edges"] == 1


def test_missing_prediction_files_produce_clear_error(tmp_path: Path) -> None:
    """Test 7: Missing prediction files produce a clear, actionable error."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="no '\\*_predictions\\.jsonl' files found"):
        build_graph_from_saved_predictions(predictions_dir=empty_dir)


def test_cached_llm_responses_reused_without_network(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test 8: Cached LLM responses are reused without network requests."""
    cache_path = tmp_path / "cache.jsonl"
    client = GeminiLLMClient(api_key="TEST_KEY", cache_path=str(cache_path), model_id="gemini-test")

    sys_prompt = "System instructions"
    user_prompt = "User prompt"
    pair_id = "TEST:PAIR:01"
    key = compute_cache_key("gemini-test", sys_prompt + user_prompt, pair_id)

    # Pre-populate cache
    cached_payload = {
        "relation_class": "causal",
        "direction": "entity_1->entity_2",
        "polarity": "increases",
        "language_status": "affirmed",
        "confidence": 0.99,
        "reasoning": "Pre-cached response",
    }
    client.cache.set(key, cached_payload)

    # Monkeypatch network to fail if accessed
    def forbidden_post(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Network call made despite cache presence!")

    monkeypatch.setattr("requests.Session.post", forbidden_post)

    response = client.generate_json(sys_prompt, user_prompt, pair_id)
    assert response == cached_payload
    stats = client.get_stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 0
    assert stats["api_calls"] == 0


def test_invalid_self_loop_pairs_filtered_and_counted() -> None:
    """Test 9: Invalid self-loop pairs are filtered and counted with diagnostic reasons."""
    builder = EvidenceGraphBuilder()
    # Both entities normalize to 'Outcome' (e.g. Type 2 Diabetes and T2DM)
    self_loop_pair = _create_test_pair(
        "PMID:600:s01:p01", "PMID:600", "PMID:600:s01", "Outcome", "Outcome", "T2DM and Type 2 diabetes"
    )
    pred = _create_test_prediction(self_loop_pair, "spaCy-Rules", pmid="600")

    builder.add_prediction(pred, self_loop_pair)

    assert builder.causal_graph.number_of_edges() == 0
    stats = builder.get_filtering_stats()
    assert stats["filtered_self_loops_count"] == 1
    assert len(builder.filtered_self_loops) == 1
    assert builder.filtered_self_loops[0]["canonical_node"] == "Outcome"
    assert builder.filtered_self_loops[0]["reason"] == "self_loop_identical_canonical_node"


def test_graph_serialization_citations_and_visualization(tmp_path: Path) -> None:
    """Test 10: Existing graph serialization, citations, and visualization still work."""
    builder = EvidenceGraphBuilder()
    pair = _create_test_pair("PMID:700:s01:p01", "PMID:700", "PMID:700:s01", "BMICat", "Outcome")
    pred = _create_test_prediction(pair, "spaCy-Rules", pmid="700")
    builder.add_prediction(pred, pair)

    out_graphml = tmp_path / "graph.graphml"
    out_prov = tmp_path / "provenance.json"
    out_png = tmp_path / "plot.png"

    export_graph_for_phase2(builder.causal_graph, out_graphml, out_prov)
    plot_evidence_graph(builder.causal_graph, out_png)

    assert out_graphml.exists()
    assert out_prov.exists()
    assert out_png.exists()
    assert out_png.stat().st_size > 0

    with open(out_prov, encoding="utf-8") as f:
        prov_data = json.load(f)
    assert len(prov_data["edges"]) == 1
    edge = prov_data["edges"][0]
    assert edge["source"] == "BMICat"
    assert edge["target"] == "Outcome"
    assert edge["unique_papers_count"] == 1
    assert edge["unique_evidence_sentences"] == 1


def test_evaluation_metrics_separate_from_graph_stats() -> None:
    """Test 11: Extractor evaluation metrics remain strictly separate from evidence graph statistics."""
    # Build minimal mock outputs
    from run_pipeline import run_full_pipeline

    # Mock extractors inside run_full_pipeline or check structure
    builder = EvidenceGraphBuilder()
    pair = _create_test_pair("PMID:800:s01:p01", "PMID:800", "PMID:800:s01", "BMICat", "Outcome")
    pred = _create_test_prediction(pair, "spaCy-Rules", pmid="800")
    builder.add_prediction(pred, pair)

    stats = builder.get_filtering_stats()
    # Graph statistics focus on edges and filtering
    assert "causal_edges_count" in stats
    assert "filtered_self_loops_count" in stats
    # Neither graph stats nor edge attributes conflate Macro-F1 with support_count
    edge = builder.causal_graph["BMICat"]["Outcome"]
    assert "macro_f1" not in edge
    assert "support_count" in edge
    assert isinstance(edge["support_count"], int)


def test_keyword_heuristic_extractor_renamed_and_runnable() -> None:
    """Test 12: The heuristic extractor is renamed to KeywordHeuristicExtractor and remains runnable."""
    assert DistilBERTExtractor is KeywordHeuristicExtractor
    extractor = KeywordHeuristicExtractor()
    assert extractor.name == "KeywordHeuristic"

    pair = _create_test_pair("TEST:01", "DOC:01", "DOC:01:s01", "BMICat", "Outcome", "Obesity causes diabetes.")
    pred = extractor.predict_single(pair)
    assert pred.validation_status == "valid"
    assert pred.label.relation_class == RelationClass.CAUSAL
    assert pred.label.reasoning == "Keyword heuristic classification: causal"
