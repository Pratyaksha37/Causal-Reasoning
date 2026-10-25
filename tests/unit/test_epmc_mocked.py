"""Unit tests for Europe PMC client with mocked HTTP responses."""

from unittest.mock import MagicMock, patch

from src.retrieval.epmc_client import EuropePMCClient


def test_epmc_client_pagination_and_exclusions() -> None:
    client = EuropePMCClient(rate_limit_per_second=0.0)

    # Page 1 response: 2 items, one missing abstract
    page1_data = {
        "nextCursorMark": "cursor_2",
        "resultList": {
            "result": [
                {
                    "title": "Study 1",
                    "abstractText": "This is abstract 1.",
                    "pmid": "111",
                    "pubYear": "2023",
                    "isOpenAccess": "Y",
                    "license": "cc by",
                    "citedByCount": 25,
                },
                {
                    "title": "Study 2 without abstract",
                    "pmid": "222",
                    "pubYear": "2023",
                    "isOpenAccess": "N",
                },
            ]
        },
    }

    # Page 2 response: 1 valid item, then end of pagination
    page2_data = {
        "nextCursorMark": "cursor_2",  # cursorMark unchanged = end
        "resultList": {
            "result": [
                {
                    "title": "Study 3",
                    "abstractText": "This is abstract 3.",
                    "pmid": "333",
                    "pubYear": "2024",
                    "isOpenAccess": "N",
                    "citedByCount": 5,
                }
            ]
        },
    }

    mock_response_1 = MagicMock()
    mock_response_1.raise_for_status = MagicMock()
    mock_response_1.json = MagicMock(return_value=page1_data)

    mock_response_2 = MagicMock()
    mock_response_2.raise_for_status = MagicMock()
    mock_response_2.json = MagicMock(return_value=page2_data)

    with patch.object(client.session, "get", side_effect=[mock_response_1, mock_response_2]):
        docs, exclusions = client.retrieve_corpus(
            query="test query",
            max_results=10,
            page_size=2,
        )

        assert len(docs) == 2
        assert docs[0].doc_id == "PMID:111"
        assert docs[0].full_text_stored is True  # CC-BY Open Access
        assert docs[1].doc_id == "PMID:333"
        assert docs[1].full_text_stored is False  # Non-OA

        assert len(exclusions) == 1
        assert exclusions[0]["doc_id"] == "222"
        assert "No abstract available" in exclusions[0]["reason"]
