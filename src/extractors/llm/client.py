"""Gemini / OpenAI-compatible REST API client with caching, structured JSON mode, and retries."""

import json
import logging
import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.extractors.llm.cache import LLMCache, compute_cache_key

logger = logging.getLogger(__name__)


class GeminiLLMClient:
    """Client for querying Google Gemini (OpenAI-compatible) endpoint for relation extraction."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai",
        model_id: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 5,
        timeout: int = 30,
        cache_path: str = "data/cache/llm_cache.jsonl",
    ) -> None:
        load_dotenv()
        candidate_key = api_key.strip() if (api_key and not api_key.startswith("$")) else None
        self.api_key = (
            candidate_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        )
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id or os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite"
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache = LLMCache(cache_file=cache_path)
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.api_calls: int = 0
        self.failed_calls: int = 0

        self.session = requests.Session()
        retries = Retry(
            total=max_retries,
            backoff_factor=1.5,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_stats(self) -> dict[str, int]:
        """Return cache hits, misses, API calls, and failure statistics."""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "api_calls": self.api_calls,
            "failed_calls": self.failed_calls,
        }

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        pair_id: str = "",
    ) -> dict[str, Any]:
        """
        Query Gemini model requesting structured JSON output with cache checking and retries.

        Args:
            system_prompt: Guidelines and JSON schema instructions.
            user_prompt: Sentence and marked entity pair information.
            pair_id: Candidate pair ID for cache keying and auditability.

        Returns:
            Dict parsed from JSON response.
        """
        cache_key = compute_cache_key(self.model_id, system_prompt + user_prompt, pair_id)
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            logger.debug("Cache hit for pair %s", pair_id)
            return cached  # type: ignore[no-any-return]

        self.cache_misses += 1

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment or config. Please set GEMINI_API_KEY in .env."
            )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "DiabetesCausalNLP-GeminiClient/1.0",
        }

        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }

        # Attempt request with rate-limit retry loop
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                self.api_calls += 1
                response = self.session.post(
                    endpoint, headers=headers, json=payload, timeout=self.timeout
                )
                if response.status_code == 429:
                    err_text = response.text
                    if "limit exceeded on Free Tier" in err_text or "GenerateRequestsPerDay" in err_text:
                        if self.model_id != "gemini-3.1-flash-lite":
                            old_model = self.model_id
                            self.model_id = "gemini-3.1-flash-lite"
                            logger.info(
                                "Free daily quota reached on %s; rotating to %s...",
                                old_model,
                                self.model_id,
                            )
                            payload["model"] = self.model_id
                            time.sleep(1.0)
                            continue

                    header_wait = response.headers.get("Retry-After")
                    if header_wait:
                        sleep_time = max(float(header_wait), 5.0)
                    else:
                        sleep_time = 5.0 * (attempt + 1)

                    logger.warning(
                        "Rate limit reached (HTTP 429). Retrying in %.1fs (attempt %d/%d)...",
                        sleep_time,
                        attempt + 1,
                        self.max_retries,
                    )
                    time.sleep(sleep_time)
                    continue

                response.raise_for_status()
                data = response.json()
                content_str = data["choices"][0]["message"]["content"].strip()

                # Clean markdown fences if present
                if content_str.startswith("```"):
                    content_str = re.sub(r"^```(?:json)?\s*", "", content_str)
                    content_str = re.sub(r"\s*```$", "", content_str)

                parsed = json.loads(content_str)
                self.cache.set(cache_key, parsed)

                # Gentle pacing delay between uncached API calls to stay within free-tier limits
                time.sleep(1.5)
                return parsed  # type: ignore[no-any-return]

            except Exception as e:
                self.failed_calls += 1
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(2.0 * (attempt + 1))

        if last_error:
            # Mask any possible accidental key inclusion
            sanitized_err = str(last_error).replace(self.api_key, "***") if self.api_key else str(last_error)
            raise RuntimeError(f"API request failed: {sanitized_err}") from None
        raise RuntimeError("Failed to generate response after retries.")


# Alias for backward compatibility
GroqLLMClient = GeminiLLMClient
