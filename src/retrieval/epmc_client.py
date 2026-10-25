"""Europe PMC REST API client with pagination, exponential backoff, and rate limiting."""

import datetime
import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.retrieval.licence import assess_licence
from src.schemas.document import Document

logger = logging.getLogger(__name__)


class EuropePMCClient:
    """Client for retrieving publication metadata and abstracts from Europe PMC."""

    def __init__(
        self,
        base_url: str = "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        timeout: int = 30,
        max_retries: int = 5,
        backoff_factor: float = 1.5,
        rate_limit_per_second: float = 5.0,
        email: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.min_interval = 1.0 / rate_limit_per_second if rate_limit_per_second > 0 else 0.0
        self.last_request_time = 0.0

        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        user_agent = f"DiabetesCausalNLP/1.0 (mailto:{email or 'research@example.com'})"
        self.session.headers.update({"User-Agent": user_agent})

    def _throttle(self) -> None:
        """Enforce inter-request rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def search_page(
        self,
        query: str,
        cursor_mark: str = "*",
        page_size: int = 100,
        sort_by: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a single page of results using Europe PMC cursorMark pagination."""
        self._throttle()
        params: dict[str, Any] = {
            "query": query,
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor_mark,
            "resultType": "core",
        }
        if sort_by:
            params["sort"] = sort_by

        response = self.session.get(self.base_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}

    def retrieve_corpus(
        self,
        query: str,
        max_results: int = 500,
        page_size: int = 100,
        sort_by: str | None = None,
    ) -> tuple[list[Document], list[dict[str, str]]]:
        """
        Retrieve publications matching query up to max_results.

        Returns:
            Tuple of (valid_documents, exclusions).
        """
        documents: list[Document] = []
        exclusions: list[dict[str, str]] = []
        cursor_mark = "*"
        retrieval_date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")

        logger.info("Starting Europe PMC retrieval for query: %s (limit: %d)", query, max_results)

        while len(documents) < max_results:
            batch_size = min(page_size, max_results - len(documents))
            try:
                data = self.search_page(
                    query=query, cursor_mark=cursor_mark, page_size=batch_size, sort_by=sort_by
                )
            except Exception as e:
                logger.error("Failed fetching page with cursorMark %s: %s", cursor_mark, e)
                break

            result_list = data.get("resultList", {}).get("result", [])
            next_cursor = data.get("nextCursorMark")

            if not result_list:
                logger.info("No more results returned from Europe PMC API.")
                break

            for item in result_list:
                title = item.get("title", "").strip()
                abstract = item.get("abstractText", "").strip()
                pmid = item.get("pmid")
                pmcid = item.get("pmcid")
                doi = item.get("doi")
                pub_year = str(item.get("pubYear", ""))

                doc_identifier = pmid or pmcid or doi or title[:30]

                if not title:
                    exclusions.append(
                        {
                            "doc_id": doc_identifier,
                            "title": "",
                            "reason": "Missing article title",
                        }
                    )
                    continue

                if not abstract:
                    exclusions.append(
                        {
                            "doc_id": doc_identifier,
                            "title": title,
                            "reason": "No abstract available in Europe PMC core metadata",
                        }
                    )
                    continue

                licence_info = assess_licence(item)
                cited_by = int(item.get("citedByCount", 0))

                doc = Document.create(
                    title=title,
                    abstract_text=abstract,
                    pub_date=pub_year,
                    query_provenance=query,
                    retrieval_date=retrieval_date,
                    pmid=pmid,
                    pmcid=pmcid,
                    doi=doi,
                    licence_note=licence_info["licence_note"],
                    cited_by_count=cited_by,
                    full_text_stored=licence_info["can_store_full_text"],
                )
                documents.append(doc)

                if len(documents) >= max_results:
                    break

            if next_cursor == cursor_mark or not next_cursor:
                logger.info("Reached end of result set (cursorMark unchanged).")
                break
            cursor_mark = next_cursor

        logger.info("Retrieved %d valid documents, %d excluded.", len(documents), len(exclusions))
        return documents, exclusions
