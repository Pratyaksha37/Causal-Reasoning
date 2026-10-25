"""Unit tests for mapping report statistics and unmapped tracking."""

from src.mapping.report import generate_mapping_report
from src.schemas.text import Mention


def test_generate_mapping_report() -> None:
    mentions = [
        Mention(
            mention_id="m1",
            sent_id="s1",
            doc_id="d1",
            surface_text="physical activity",
            start_char=0,
            end_char=17,
            canonical_node_id="PhysActivity",
        ),
        Mention(
            mention_id="m2",
            sent_id="s1",
            doc_id="d1",
            surface_text="physical inactivity",
            start_char=20,
            end_char=39,
            canonical_node_id="PhysActivity",
            polarity_flip=True,
        ),
        Mention(
            mention_id="m3",
            sent_id="s1",
            doc_id="d1",
            surface_text="type 2 diabetes",
            start_char=50,
            end_char=65,
            canonical_node_id="Outcome",
        ),
        Mention(
            mention_id="m4",
            sent_id="s1",
            doc_id="d1",
            surface_text="smoking",
            start_char=70,
            end_char=77,
            canonical_node_id="unmapped",
        ),
        Mention(
            mention_id="m5",
            sent_id="s1",
            doc_id="d1",
            surface_text="smoking",
            start_char=80,
            end_char=87,
            canonical_node_id="unmapped",
        ),
    ]

    report = generate_mapping_report(mentions)
    assert report["total_mentions"] == 5
    assert report["mapped_count"] == 3
    assert report["unmapped_count"] == 2
    assert report["mapped_rate"] == 0.6
    assert report["unmapped_rate"] == 0.4
    assert report["polarity_flipped_count"] == 1
    assert report["canonical_breakdown"]["PhysActivity"] == 2
    assert report["canonical_breakdown"]["Outcome"] == 1
    assert report["top_unmapped_phrases"][0]["phrase"] == "smoking"
    assert report["top_unmapped_phrases"][0]["count"] == 2
