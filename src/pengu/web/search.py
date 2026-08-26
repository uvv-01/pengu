"""
Web search provider abstraction for Pengu.

Uses DuckDuckGo (free, no API key required) as the default search provider.
Falls back gracefully if unavailable.

Design principle: Never make web search mandatory.
Pengu works without internet.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.web.search")


@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "rank": self.rank,
        }


class WebSearchProvider(ABC):
    """Base class for web search providers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._available = False
        self._last_error = ""

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Search the web."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the search provider is available."""
        ...

    def is_available(self) -> bool:
        """Quick check without network call."""
        return self._available

    @abstractmethod
    async def fetch(self, url: str, max_chars: int = 10000) -> Optional[str]:
        """Fetch content from a URL."""
        ...


class DuckDuckGoProvider(WebSearchProvider):
    """
    DuckDuckGo search provider.

    Uses the duckduckgo_search library (free, no API key).
    License: MIT
    """

    def __init__(self) -> None:
        super().__init__(name="duckduckgo")

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Search DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for i, r in enumerate(ddgs.text(query, max_results=max_results)):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        rank=i + 1,
                    ))

            logger.info("web_search_completed", query=query, results=len(results))
            return results

        except ImportError:
            logger.warning("duckduckgo_not_installed")
            self._available = False
            self._last_error = "duckduckgo_search not installed"
            return []
        except Exception as e:
            logger.warning("web_search_failed", error=str(e))
            self._available = False
            self._last_error = str(e)
            return []

    async def health_check(self) -> bool:
        """Check if DuckDuckGo search is available."""
        try:
            from duckduckgo_search import DDGS
            # Quick test search
            with DDGS() as ddgs:
                results = list(ddgs.text("test", max_results=1))
            self._available = True
            self._last_error = ""
            return True
        except ImportError:
            self._available = False
            self._last_error = "duckduckgo_search not installed"
            return False
        except Exception as e:
            self._available = False
            self._last_error = str(e)
            return False

    async def fetch(self, url: str, max_chars: int = 10000) -> Optional[str]:
        """Fetch content from a URL using httpx."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, follow_redirects=True)
                if resp.status_code == 200:
                    return resp.text[:max_chars]
            return None
        except Exception as e:
            logger.warning("web_fetch_failed", url=url, error=str(e))
            return None


class MockSearchProvider(WebSearchProvider):
    """Mock search provider for testing."""

    def __init__(self) -> None:
        super().__init__(name="mock")
        self._available = True
        self._results: list[SearchResult] = []

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchResult]:
        return self._results[:max_results]

    async def health_check(self) -> bool:
        return True

    async def fetch(self, url: str, max_chars: int = 10000) -> Optional[str]:
        return f"Mock content for {url}"

    def set_results(self, results: list[SearchResult]) -> None:
        self._results = results


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_search_provider: Optional[WebSearchProvider] = None


def get_search_provider() -> WebSearchProvider:
    """Get the web search provider (lazy initialization)."""
    global _search_provider
    if _search_provider is None:
        _search_provider = DuckDuckGoProvider()
    return _search_provider


def reset_search_provider() -> WebSearchProvider:
    """Reset the search provider (for testing)."""
    global _search_provider
    _search_provider = DuckDuckGoProvider()
    return _search_provider
