"""Unit and regression tests for graph coverage fixes, trigger precision, and association provenance."""

from pathlib import Path
import tempfile
import networkx as nx
import pytest

from src.extractors.distilbert_extractor import KeywordHeuristicExtractor
from src.extractors.rules_extractor import RulesExtractor
from src.graph.builder import EvidenceGraphBuilder
from src.graph.exporter import export_association_graph, export_graph_for_phase2
from src.graph.visualizer import plot_combined_evidence_graph, plot_evidence_graph
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


def _make_mention(surface: str, node_id: str, start: int = 0, end: int = 5) -> Mention:
    return Mention(
        mention_id=f"m_{node_id}_{start}",
        sent_id="s1",
        doc_id="DOC1",
        surface_text=surface,
        start_char=start,
        end_char=end,
        canonical_node_id=node_id,
    )


def test_all_cause_mortality_not_classified_as_causal() -> None:
    """Ensure compound phrases like 'all-cause mortality' or 'because' do not trigger false causal claims."""
    rules_ext = RulesExtractor()
    heuristic_ext = KeywordHeuristicExtractor()

    pair = CandidatePair(
        pair_id="TEST:1",
        doc_id="PMID:10001",
        sent_id="s1",
        sentence_text="In both groups, no statistically significant association was found with all-cause mortality.",
        entity_1=_make_mention("T2DM", "Outcome", 0, 4),
        entity_2=_make_mention("men", "Sex", 10, 13),
    )

    p_rules = rules_ext.predict_single(pair)
    p_heuristic = heuristic_ext.predict_single(pair)

    assert p_rules.label.relation_class != RelationClass.CAUSAL, "all-cause mortality falsely triggered causal in rules"
    assert p_heuristic.label.relation_class != RelationClass.CAUSAL, "all-cause mortality falsely triggered causal in heuristic"


def test_because_substring_not_classified_as_causal() -> None:
    """Ensure the word 'because' does not match 'cause' substring in KeywordHeuristicExtractor."""
    heuristic_ext = KeywordHeuristicExtractor()
    pair = CandidatePair(
        pair_id="TEST:2",
        doc_id="PMID:10002",
        sent_id="s1",
        sentence_text="The study was paused because sample sizes were inadequate.",
        entity_1=_make_mention("obesity", "BMICat", 0, 7),
        entity_2=_make_mention("diabetes", "Outcome", 10, 18),
    )
    p_heuristic = heuristic_ext.predict_single(pair)
    assert p_heuristic.label.relation_class == RelationClass.NONE


def test_association_graph_retains_full_provenance_and_unique_counts() -> None:
    """Test that association edges track unique sentences, papers, extractors, and full evidence."""
    builder = EvidenceGraphBuilder()

    pair1 = CandidatePair(
        pair_id="P1",
        doc_id="PMID:20001",
        sent_id="s1",
        sentence_text="Short sleep duration was positively associated with insulin resistance in adults.",
        entity_1=_make_mention("sleep deprivation", "SleepDuration", 0, 17),
        entity_2=_make_mention("insulin resistance", "InsulinResistance", 52, 70),
    )

    pred1 = Prediction(
        pair_id="P1",
        extractor_name="spaCy-Rules",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.ASSOCIATION_ONLY,
            direction=Direction.NOT_APPLICABLE,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.80,
            provenance=Provenance(doc_id="PMID:20001", sent_id="s1", pmid="20001"),
        ),
    )

    # Second extractor predicting on the exact same sentence
    pred2 = Prediction(
        pair_id="P1",
        extractor_name="Prompted-LLM-Gemini",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.ASSOCIATION_ONLY,
            direction=Direction.NOT_APPLICABLE,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.90,
            provenance=Provenance(doc_id="PMID:20001", sent_id="s1", pmid="20001"),
        ),
    )

    # Third prediction on a DIFFERENT sentence and different paper
    pair2 = CandidatePair(
        pair_id="P2",
        doc_id="PMID:20002",
        sent_id="s4",
        sentence_text="HOMA-IR was correlated with poor sleep quality.",
        entity_1=_make_mention("HOMA-IR", "InsulinResistance", 0, 7),
        entity_2=_make_mention("poor sleep quality", "SleepDuration", 28, 46),
    )
    pred3 = Prediction(
        pair_id="P2",
        extractor_name="KeywordHeuristic",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.ASSOCIATION_ONLY,
            direction=Direction.NOT_APPLICABLE,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.82,
            provenance=Provenance(doc_id="PMID:20002", sent_id="s4", pmid="20002"),
        ),
    )

    builder.add_prediction(pred1, pair1)
    builder.add_prediction(pred2, pair1)
    builder.add_prediction(pred3, pair2)

    assoc_g = builder.association_graph
    assert assoc_g.has_edge("InsulinResistance", "SleepDuration")
    edge_data = assoc_g["InsulinResistance"]["SleepDuration"]

    # Should count 2 unique sentences (not 3 predictions)
    assert edge_data["support_count"] == 2
    assert edge_data["unique_evidence_sentences"] == 2
    # Should count 2 unique papers
    assert edge_data["unique_papers_count"] == 2
    # Supporting extractors list
    assert sorted(edge_data["extractors"]) == ["KeywordHeuristic", "Prompted-LLM-Gemini", "spaCy-Rules"]
    # Full evidence records
    assert len(edge_data["evidence"]) == 2


def test_get_combined_graph_multi_relational() -> None:
    """Test get_combined_graph integrates both causal DAG and association network."""
    builder = EvidenceGraphBuilder()

    # Add causal edge
    pair_causal = CandidatePair(
        pair_id="C1",
        doc_id="PMID:30001",
        sent_id="s1",
        sentence_text="Obesity leads to type 2 diabetes mellitus.",
        entity_1=_make_mention("obesity", "BMICat", 0, 7),
        entity_2=_make_mention("type 2 diabetes mellitus", "Outcome", 17, 42),
    )
    pred_causal = Prediction(
        pair_id="C1",
        extractor_name="RulesExtractor",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.CAUSAL,
            direction=Direction.E1_TO_E2,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.95,
        ),
    )
    builder.add_prediction(pred_causal, pair_causal)

    # Add association edge
    pair_assoc = CandidatePair(
        pair_id="A1",
        doc_id="PMID:30002",
        sent_id="s2",
        sentence_text="Hypertension is associated with type 2 diabetes mellitus.",
        entity_1=_make_mention("hypertension", "Hypertension", 0, 12),
        entity_2=_make_mention("type 2 diabetes mellitus", "Outcome", 31, 56),
    )
    pred_assoc = Prediction(
        pair_id="A1",
        extractor_name="RulesExtractor",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.ASSOCIATION_ONLY,
            direction=Direction.NOT_APPLICABLE,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.85,
        ),
    )
    builder.add_prediction(pred_assoc, pair_assoc)

    combined_g = builder.get_combined_graph()
    assert combined_g.has_edge("BMICat", "Outcome")
    assert combined_g["BMICat"]["Outcome"]["relation_class"] == "causal"

    assert combined_g.has_edge("Hypertension", "Outcome") or combined_g.has_edge("Outcome", "Hypertension")
    u, v = ("Hypertension", "Outcome") if combined_g.has_edge("Hypertension", "Outcome") else ("Outcome", "Hypertension")
    assert combined_g[u][v]["relation_class"] == "association_only"


def test_export_and_plot_artifacts_completeness() -> None:
    """Test exporting GraphML, provenance JSON, and generating visualization."""
    builder = EvidenceGraphBuilder()

    pair = CandidatePair(
        pair_id="EXP:1",
        doc_id="PMID:40001",
        sent_id="s1",
        sentence_text="Physical inactivity increases risk of type 2 diabetes mellitus.",
        entity_1=_make_mention("physical inactivity", "PhysActivity", 0, 19),
        entity_2=_make_mention("type 2 diabetes mellitus", "Outcome", 38, 63),
    )
    pair.entity_1.polarity_flip = True

    pred = Prediction(
        pair_id="EXP:1",
        extractor_name="Prompted-LLM-Gemini",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.CAUSAL,
            direction=Direction.E1_TO_E2,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.92,
            provenance=Provenance(doc_id="PMID:40001", sent_id="s1", pmid="40001"),
        ),
    )
    builder.add_prediction(pred, pair)

    pair_assoc = CandidatePair(
        pair_id="EXP:2",
        doc_id="PMID:40002",
        sent_id="s3",
        sentence_text="Dietary patterns are linked with physical activity levels.",
        entity_1=_make_mention("dietary pattern", "Diet", 0, 15),
        entity_2=_make_mention("physical activity", "PhysActivity", 33, 50),
    )
    pred_assoc = Prediction(
        pair_id="EXP:2",
        extractor_name="KeywordHeuristic",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.ASSOCIATION_ONLY,
            direction=Direction.NOT_APPLICABLE,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.80,
            provenance=Provenance(doc_id="PMID:40002", sent_id="s3", pmid="40002"),
        ),
    )
    builder.add_prediction(pred_assoc, pair_assoc)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        causal_graphml = tmp_path / "causal.graphml"
        assoc_graphml = tmp_path / "assoc.graphml"
        prov_json = tmp_path / "prov.json"
        fig_path = tmp_path / "graph.png"

        export_graph_for_phase2(
            graph=builder.causal_graph,
            output_graphml=causal_graphml,
            output_provenance_json=prov_json,
            association_graph=builder.association_graph,
        )
        export_association_graph(
            graph=builder.association_graph,
            output_graphml=assoc_graphml,
        )

        assert causal_graphml.exists() and causal_graphml.stat().st_size > 0
        assert assoc_graphml.exists() and assoc_graphml.stat().st_size > 0
        assert prov_json.exists() and prov_json.stat().st_size > 0

        # Plotting
        plot_combined_evidence_graph(
            causal_graph=builder.causal_graph,
            association_graph=builder.association_graph,
            output_path=fig_path,
        )
        assert fig_path.exists() and fig_path.stat().st_size > 1000


def test_unrelated_endpoint_not_accepted_as_causal_for_diabetes() -> None:
    """Ensure causality for an unrelated endpoint (e.g. CTS) is not assigned to diabetes Outcome."""
    builder = EvidenceGraphBuilder()
    pair = CandidatePair(
        pair_id="PMID:42658720:s02:p01",
        doc_id="PMID:42658720",
        sent_id="s02",
        sentence_text="We replicate the result that increases in measures of obesity are causal for CTS, as well as several metabolic factors, such as obesity and type 2 diabetes.",
        entity_1=_make_mention("obesity", "BMICat", 0, 7),
        entity_2=_make_mention("type 2 diabetes", "Outcome", 50, 65),
    )
    pred = Prediction(
        pair_id=pair.pair_id,
        extractor_name="KeywordHeuristic",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.CAUSAL,
            direction=Direction.E1_TO_E2,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.88,
            provenance=Provenance(doc_id=pair.doc_id, sent_id=pair.sent_id),
        ),
    )
    builder.add_prediction(pred, pair)

    # Must NOT be added to causal DAG
    assert not builder.causal_graph.has_edge("BMICat", "Outcome")
    # Must be preserved in observational association graph
    assert builder.association_graph.has_edge("BMICat", "Outcome")


def test_parallel_trend_and_morphometry_not_accepted_as_causal() -> None:
    """Ensure ecological parallel trends and brain morphometry findings do not form causal edges."""
    builder = EvidenceGraphBuilder()
    pair_trend = CandidatePair(
        pair_id="TREND:1",
        doc_id="PMID:42503624",
        sent_id="s05",
        sentence_text="All registries showed a temporally parallel increase in the incidence of pediatric obesity and youth-onset T2DM.",
        entity_1=_make_mention("obesity", "BMICat", 0, 7),
        entity_2=_make_mention("T2DM", "Outcome", 20, 24),
    )
    pred_trend = Prediction(
        pair_id=pair_trend.pair_id,
        extractor_name="KeywordHeuristic",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.CAUSAL,
            direction=Direction.E1_TO_E2,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.88,
            provenance=Provenance(doc_id=pair_trend.doc_id, sent_id=pair_trend.sent_id),
        ),
    )
    builder.add_prediction(pred_trend, pair_trend)

    pair_morph = CandidatePair(
        pair_id="MORPH:1",
        doc_id="PMID:42571999",
        sent_id="s04",
        sentence_text="Furthermore, SBM captured localized morphometric increases in the fusiform gyrus unique to obese T2DM patients.",
        entity_1=_make_mention("obese", "BMICat", 0, 5),
        entity_2=_make_mention("T2DM", "Outcome", 10, 14),
    )
    pred_morph = Prediction(
        pair_id=pair_morph.pair_id,
        extractor_name="KeywordHeuristic",
        extractor_version="1.0.0",
        label=RelationLabel(
            relation_class=RelationClass.CAUSAL,
            direction=Direction.E1_TO_E2,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.88,
            provenance=Provenance(doc_id=pair_morph.doc_id, sent_id=pair_morph.sent_id),
        ),
    )
    builder.add_prediction(pred_morph, pair_morph)

    assert not builder.causal_graph.has_edge("BMICat", "Outcome")
    assert builder.association_graph.has_edge("BMICat", "Outcome")

