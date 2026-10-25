"""Unit tests for sentence splitting and character offset tracking."""

from src.text.splitter import SentenceSplitter


def test_sentence_splitter_basic() -> None:
    splitter = SentenceSplitter()
    text = "Physical inactivity increases the risk of T2DM. Regular exercise reduces diabetes incidence."
    sentences = splitter.split(doc_id="PMID:100", text=text)

    assert len(sentences) == 2
    assert sentences[0].sent_id == "PMID:100:s01"
    assert sentences[1].sent_id == "PMID:100:s02"
    assert sentences[0].text == "Physical inactivity increases the risk of T2DM."
    assert sentences[1].text == "Regular exercise reduces diabetes incidence."

    # Test exact substring offsets match
    for s in sentences:
        assert text[s.start_char : s.end_char] == s.text


def test_sentence_splitter_empty() -> None:
    splitter = SentenceSplitter()
    assert splitter.split("PMID:0", "") == []
    assert splitter.split("PMID:0", "   \n  ") == []
