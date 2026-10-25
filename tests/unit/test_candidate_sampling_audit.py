"""Regression and audit tests for candidate sampling, causal filters, and benchmark protocol."""

import json
from pathlib import Path

from src.extractors.distilbert_extractor import KeywordHeuristicExtractor
from src.extractors.rules_extractor import RulesExtractor
from src.graph.builder import EvidenceGraphBuilder
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


def test_full_corpus_candidate_selection_includes_later_records() -> None:
    """Ensure candidate selection over candidates_corpus includes later lines such as PMIDs 41424208 and 42415555."""
    corpus_path = Path("data/candidates_corpus.jsonl")
    assert corpus_path.exists()

    all_non_self: list[CandidatePair] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = CandidatePair.model_validate_json(line)
                if p.entity_1.canonical_node_id != p.entity_2.canonical_node_id:
                    all_non_self.append(p)

    assert len(all_non_self) == 1143
    pmids = {p.doc_id for p in all_non_self}
    assert "PMID:41424208" in pmids
    assert "PMID:42415555" in pmids


def test_valid_causal_evidence_accepted_into_causal_dag() -> None:
    """Test that legitimate causal evidence from Mendelian Randomization and longitudinal etiology is accepted."""
    p1 = CandidatePair(
        pair_id="PMID:41424208:s12:p01",
        doc_id="PMID:41424208",
        sent_id="PMID:41424208:s12",
        sentence_text="Crucially, MR analysis identifies abdominal obesity, quantified by waist circumference, as the dominant causal link between NAFLD and T2DM.",
        entity_1=Mention(
            mention_id="m1", sent_id="s12", doc_id="41424208", surface_text="obesity",
            start_char=44, end_char=51, entity_type="CONCEPT", canonical_node_id="BMICat",
            alias_matched="obesity", polarity_flip=False
        ),
        entity_2=Mention(
            mention_id="m2", sent_id="s12", doc_id="41424208", surface_text="T2DM",
            start_char=134, end_char=138, entity_type="CONCEPT", canonical_node_id="Outcome",
            alias_matched="t2dm", polarity_flip=False
        ),
    )

    p2 = CandidatePair(
        pair_id="PMID:42415555:s01:p02",
        doc_id="PMID:42415555",
        sent_id="PMID:42415555:s01",
        sentence_text="Pediatric obesity is a growing global concern because it leads to an increased prevalence of abnormal glucose metabolism, including dysglycemia, prediabetes, and type 2 diabetes.",
        entity_1=Mention(
            mention_id="m1", sent_id="s01", doc_id="42415555", surface_text="obesity",
            start_char=10, end_char=17, entity_type="CONCEPT", canonical_node_id="BMICat",
            alias_matched="obesity", polarity_flip=False
        ),
        entity_2=Mention(
            mention_id="m2", sent_id="s01", doc_id="42415555", surface_text="type 2 diabetes",
            start_char=159, end_char=174, entity_type="CONCEPT", canonical_node_id="Outcome",
            alias_matched="type 2 diabetes", polarity_flip=False
        ),
    )

    rules = RulesExtractor()
    heur = KeywordHeuristicExtractor()

    preds1 = rules.predict([p1, p2])
    preds2 = heur.predict([p1, p2])

    builder = EvidenceGraphBuilder()
    builder.add_prediction(preds1[0], p1)
    builder.add_prediction(preds2[0], p1)
    builder.add_prediction(preds1[1], p2)
    builder.add_prediction(preds2[1], p2)

    # Must produce BMICat -> Outcome in the causal DAG
    assert builder.causal_graph.has_edge("BMICat", "Outcome")
    edge_data = builder.causal_graph.get_edge_data("BMICat", "Outcome")
    assert edge_data["polarity"] == "increases"
    assert edge_data["unique_papers_count"] == 2
    assert edge_data["unique_evidence_sentences"] == 2
    assert "spaCy-Rules" in edge_data["extractors"]
    assert "KeywordHeuristic" in edge_data["extractors"]


def test_unrelated_endpoints_and_trends_routed_to_association() -> None:
    """Test that unrelated endpoints (CTS, IVDD, case reports, trends) are rejected from the DAG."""
    test_cases = [
        ("PMID:901", "Causal role in increasing the risk of CTS in patients with type 2 diabetes."),
        ("PMID:902", "Potential causal links between multi-domain factors and IVDD."),
        ("PMID:903", "A temporally parallel increase in the incidence of obesity and diabetes was observed."),
        ("PMID:904", "Morphometric increases in fusiform gyrus volume were noted in diabetes patients."),
        ("PMID:905", "We report the case of an obese patient who developed marked hepatocellular injury."),
    ]

    builder = EvidenceGraphBuilder()
    for doc_id, text in test_cases:
        pair = CandidatePair(
            pair_id=f"{doc_id}:p01",
            doc_id=doc_id,
            sent_id=f"{doc_id}:s01",
            sentence_text=text,
            entity_1=Mention(
                mention_id="m1", sent_id="s01", doc_id=doc_id, surface_text="obesity",
                start_char=0, end_char=7, entity_type="CONCEPT", canonical_node_id="BMICat",
                alias_matched="obesity", polarity_flip=False
            ),
            entity_2=Mention(
                mention_id="m2", sent_id="s01", doc_id=doc_id, surface_text="diabetes",
                start_char=10, end_char=18, entity_type="CONCEPT", canonical_node_id="Outcome",
                alias_matched="diabetes", polarity_flip=False
            ),
        )
        pred = Prediction(
            pair_id=pair.pair_id,
            extractor_name="spaCy-Rules",
            extractor_version="1.0.0",
            validation_status="valid",
            label=RelationLabel(
                relation_class=RelationClass.CAUSAL,
                direction=Direction.E1_TO_E2,
                polarity=Polarity.INCREASES,
                language_status=LanguageStatus.AFFIRMED,
                confidence=0.85,
                reasoning="test",
                provenance=Provenance(
                    doc_id=doc_id,
                    sent_id=f"{doc_id}:s01",
                    sentence_text=text,
                    char_spans=[0, 18],
                    extractor_name="spaCy-Rules",
                    confidence=0.85,
                    review_status=ReviewStatus.UNREVIEWED,
                ),
            ),
        )
        builder.add_prediction(pred, pair)

    # None of these should enter the causal DAG
    assert not builder.causal_graph.has_edge("BMICat", "Outcome")
    # All must be routed to the observational association graph to preserve empirical provenance
    assert builder.association_graph.has_edge("BMICat", "Outcome")


def test_annotation_sample_zero_self_loops_and_no_leakage() -> None:
    """Verify that data/annotations/sample.jsonl has zero self-loops and zero split overlap."""
    sample_path = Path("data/annotations/sample.jsonl")
    assert sample_path.exists()

    with open(sample_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 250

    docs_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    for r in records:
        assert r["entity_1"]["canonical_node_id"] != r["entity_2"]["canonical_node_id"]
        assert r["review_status"] in ("PROVISIONAL_RULE_PSEUDO_GOLD", "NEEDS_EXPERT_ANNOTATION", "NEEDS_HUMAN_ANNOTATION")
        assert r["human_reviewed"] is False
        assert r["relation_class"] in (None, "causal", "association_only", "none")
        docs_by_split[r["split"]].add(r["doc_id"])

    # Document-level holdout integrity (zero leakage)
    assert len(docs_by_split["train"] & docs_by_split["dev"]) == 0
    assert len(docs_by_split["train"] & docs_by_split["test"]) == 0
    assert len(docs_by_split["dev"] & docs_by_split["test"]) == 0


def test_dementia_risk_in_t2dm_patients_rejected_from_causal_dag() -> None:
    """
    PMID:42828458:s04 describes dementia risk in patients who already have T2DM.
    Must NOT justify InsulinResistance -> Outcome in the causal DAG; must route to association.
    """
    p = CandidatePair(
        pair_id="PMID:42828458:s04:p01",
        doc_id="PMID:42828458",
        sent_id="PMID:42828458:s04",
        sentence_text="Multiple studies have investigated them, suggesting that insulin resistance, hyperglycemia and neuroinflammation are some potential factors which can increase risk of dementia in patients with T2DM.",
        entity_1=Mention(
            mention_id="m1", sent_id="s04", doc_id="42828458", surface_text="insulin resistance",
            start_char=58, end_char=76, entity_type="CONCEPT", canonical_node_id="InsulinResistance",
            alias_matched="insulin resistance", polarity_flip=False
        ),
        entity_2=Mention(
            mention_id="m2", sent_id="s04", doc_id="42828458", surface_text="T2DM",
            start_char=194, end_char=198, entity_type="CONCEPT", canonical_node_id="Outcome",
            alias_matched="t2dm", polarity_flip=False
        ),
    )

    pred = Prediction(
        pair_id=p.pair_id,
        extractor_name="KeywordHeuristic",
        extractor_version="1.0.0",
        validation_status="valid",
        label=RelationLabel(
            relation_class=RelationClass.CAUSAL,
            direction=Direction.E1_TO_E2,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.88,
            reasoning="test",
        ),
    )

    builder = EvidenceGraphBuilder()
    builder.add_prediction(pred, p)

    # Must be excluded from causal DAG
    assert not builder.causal_graph.has_edge("InsulinResistance", "Outcome")
    # Must be routed to association graph
    assert builder.association_graph.has_edge("InsulinResistance", "Outcome")


def test_mediated_relationship_not_direct_causal_edge() -> None:
    """
    PMID:42901975:s03 describes poor dietary quality leading to T2DM through insulin resistance.
    Must NOT create a direct interventional causal edge InsulinResistance -> Outcome.
    """
    p = CandidatePair(
        pair_id="PMID:42901975:s03:p03",
        doc_id="PMID:42901975",
        sent_id="PMID:42901975:s03",
        sentence_text="poor dietary quality can increase the risk of metabolic syndrome and T2DM through increased visceral adiposity and insulin resistance, inflammation and an unhealthy microbiome",
        entity_1=Mention(
            mention_id="m1", sent_id="s03", doc_id="42901975", surface_text="insulin resistance",
            start_char=118, end_char=136, entity_type="CONCEPT", canonical_node_id="InsulinResistance",
            alias_matched="insulin resistance", polarity_flip=False
        ),
        entity_2=Mention(
            mention_id="m2", sent_id="s03", doc_id="42901975", surface_text="T2DM",
            start_char=69, end_char=73, entity_type="CONCEPT", canonical_node_id="Outcome",
            alias_matched="t2dm", polarity_flip=False
        ),
    )

    pred = Prediction(
        pair_id=p.pair_id,
        extractor_name="spaCy-Rules",
        extractor_version="1.0.0",
        validation_status="valid",
        label=RelationLabel(
            relation_class=RelationClass.CAUSAL,
            direction=Direction.E1_TO_E2,
            polarity=Polarity.INCREASES,
            language_status=LanguageStatus.AFFIRMED,
            confidence=0.85,
            reasoning="test",
        ),
    )

    builder = EvidenceGraphBuilder()
    builder.add_prediction(pred, p)

    # Must be excluded from causal DAG
    assert not builder.causal_graph.has_edge("InsulinResistance", "Outcome")
    assert builder.association_graph.has_edge("InsulinResistance", "Outcome")


def test_evaluation_metrics_detects_duplicates_and_missing() -> None:
    """Verify that evaluate_predictions detects duplicate predictions and missing predictions."""
    from src.evaluation.metrics import evaluate_predictions

    gold = [
        {"pair_id": "P1", "relation_class": "causal", "direction": "entity_1->entity_2", "polarity": "increases", "language_status": "affirmed"},
        {"pair_id": "P2", "relation_class": "association_only", "direction": "not_applicable", "polarity": "increases", "language_status": "affirmed"},
        {"pair_id": "P3", "relation_class": "none", "direction": "not_applicable", "polarity": "not_applicable", "language_status": "affirmed"},
    ]

    # Preds missing P2, with duplicate P1
    p1_a = Prediction(
        pair_id="P1", extractor_name="TestExt", extractor_version="1.0",
        label=RelationLabel(relation_class=RelationClass.CAUSAL, direction=Direction.E1_TO_E2, polarity=Polarity.INCREASES, language_status=LanguageStatus.AFFIRMED),
    )
    p1_b = Prediction(
        pair_id="P1", extractor_name="TestExt", extractor_version="1.0",
        label=RelationLabel(relation_class=RelationClass.CAUSAL, direction=Direction.E1_TO_E2, polarity=Polarity.INCREASES, language_status=LanguageStatus.AFFIRMED),
    )
    p3 = Prediction(
        pair_id="P3", extractor_name="TestExt", extractor_version="1.0",
        label=RelationLabel(relation_class=RelationClass.NONE, direction=Direction.NOT_APPLICABLE, polarity=Polarity.NOT_APPLICABLE, language_status=LanguageStatus.AFFIRMED),
    )

    report = evaluate_predictions(gold, [p1_a, p1_b, p3])
    assert report["duplicate_predictions_count"] == 1
    assert report["missing_predictions_count"] == 1
    assert report["gold_total"] == 3
    # Missing P2 was penalized in recall for association_only
    assert report["per_class"]["association_only"]["recall"] == 0.0


def test_backup_directories_never_loaded_into_evaluation() -> None:
    """Verify that backup directories are not ingested into production predictions."""
    from src.evaluation.tables import load_predictions_from_dir

    backup_dir = Path("backups/audit_pre_repair_20261025/predictions")
    if backup_dir.exists():
        # Loading from production results/predictions/ must not load backups
        preds = load_predictions_from_dir("results/predictions")
        assert "FineTuned-DistilBERT" not in preds

