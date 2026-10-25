"""Unit tests for CandidateGenerator and mention extraction."""

from src.schemas.text import Sentence
from src.text.candidates import CandidateGenerator


def test_candidate_generation_ordered_pairs() -> None:
    generator = CandidateGenerator()
    sent = Sentence(
        doc_id="PMID:100",
        sent_id="PMID:100:s01",
        sentence_idx=0,
        text="Physical inactivity and obesity increase the risk of type 2 diabetes.",
        start_char=0,
        end_char=70,
    )

    mentions = generator.extract_mentions(sent)
    # Should identify: "physical inactivity", "obesity", "type 2 diabetes"
    assert len(mentions) >= 3
    surface_texts = [m.surface_text.lower() for m in mentions]
    assert "physical inactivity" in surface_texts
    assert "obesity" in surface_texts
    assert "type 2 diabetes" in surface_texts

    pairs = generator.generate_candidate_pairs(sent, mentions=mentions, split="train")
    # 3 mentions choose 2 = 3 pairs
    assert len(pairs) == 3

    # Check pair ordering: entity_1 precedes entity_2
    for p in pairs:
        assert p.entity_1.start_char < p.entity_2.start_char
        assert p.split == "train"
        assert p.doc_id == "PMID:100"
        assert p.sent_id == "PMID:100:s01"


def test_candidate_generation_single_or_no_mentions() -> None:
    generator = CandidateGenerator()
    sent = Sentence(
        doc_id="PMID:200",
        sent_id="PMID:200:s01",
        sentence_idx=0,
        text="This sentence only mentions obesity.",
        start_char=0,
        end_char=36,
    )
    pairs = generator.generate_candidate_pairs(sent)
    assert pairs == []
