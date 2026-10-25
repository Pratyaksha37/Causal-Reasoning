"""Persistent JSONL response cache for LLM queries."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_cache_key(model_id: str, prompt_text: str, pair_id: str) -> str:
    """Compute deterministic SHA-256 cache key."""
    content = f"{model_id}::{prompt_text}::{pair_id}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class LLMCache:
    """File-backed key-value cache for LLM responses."""

    def __init__(self, cache_file: Path | str = "data/cache/llm_cache.jsonl") -> None:
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load existing cache entries into memory."""
        if not self.cache_file.exists():
            return

        with open(self.cache_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    key = record.get("key")
                    val = record.get("response")
                    if key and val:
                        self._memory_cache[key] = val
                except json.JSONDecodeError:
                    continue

        logger.debug(
            "Loaded %d cached LLM responses from %s", len(self._memory_cache), self.cache_file
        )

    def get(self, key: str) -> Any | None:
        """Retrieve cached response if available."""
        return self._memory_cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Save response to memory and append to cache file."""
        self._memory_cache[key] = value
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": value}, ensure_ascii=False) + "\n")
