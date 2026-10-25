"""Unit tests for Cohen's Kappa agreement and disagreement reporting."""

from src.annotation.kappa import compute_agreement


def test_kappa_perfect_agreement() -> None:
    # Two annotators with identical ratings across 4 pairs
    a1 = [
        {
            "pair_id": "P1",
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P2",
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "decreases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P3",
            "relation_class": "association_only",
            "direction": "not_applicable",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P4",
            "relation_class": "none",
            "direction": "not_applicable",
            "polarity": "not_applicable",
            "language_status": "affirmed",
        },
    ]
    a2 = [
        {
            "pair_id": "P1",
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P2",
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "decreases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P3",
            "relation_class": "association_only",
            "direction": "not_applicable",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P4",
            "relation_class": "none",
            "direction": "not_applicable",
            "polarity": "not_applicable",
            "language_status": "affirmed",
        },
    ]

    report = compute_agreement(a1, a2)
    assert report["total_pairs_compared"] == 4
    assert report["relation_class_kappa"] == 1.0
    assert report["raw_agreement_rate"] == 1.0
    assert report["total_disagreements"] == 0
    assert report["disagreements"] == []


def test_kappa_with_known_disagreement() -> None:
    a1 = [
        {
            "pair_id": "P1",
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P2",
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "decreases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P3",
            "relation_class": "association_only",
            "direction": "not_applicable",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P4",
            "relation_class": "none",
            "direction": "not_applicable",
            "polarity": "not_applicable",
            "language_status": "affirmed",
        },
    ]
    a2 = [
        {
            "pair_id": "P1",
            "relation_class": "causal",
            "direction": "entity_1->entity_2",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P2",
            "relation_class": "association_only",
            "direction": "not_applicable",
            "polarity": "decreases",
            "language_status": "affirmed",
        },  # Disagreed!
        {
            "pair_id": "P3",
            "relation_class": "association_only",
            "direction": "not_applicable",
            "polarity": "increases",
            "language_status": "affirmed",
        },
        {
            "pair_id": "P4",
            "relation_class": "none",
            "direction": "not_applicable",
            "polarity": "not_applicable",
            "language_status": "affirmed",
        },
    ]

    report = compute_agreement(a1, a2)
    assert report["total_pairs_compared"] == 4
    assert report["raw_agreement_rate"] == 0.75
    assert report["relation_class_kappa"] < 1.0
    assert report["total_disagreements"] == 1
    assert report["disagreements"][0]["pair_id"] == "P2"
    assert report["disagreements"][0]["annotator_1"]["relation_class"] == "causal"
    assert report["disagreements"][0]["annotator_2"]["relation_class"] == "association_only"
