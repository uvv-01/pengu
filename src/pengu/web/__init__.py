"""
Web search and browser interface for Pengu.

Provides:
  - WebSearchProvider: abstraction for web search
  - BrowserInterface: abstraction for browser automation

All providers are optional. Pengu works without internet.
"""

from __future__ import annotations

from pengu.web.search import WebSearchProvider, SearchResult, get_search_provider
from pengu.web.browser import BrowserInterface, get_browser

__all__ = [
    "WebSearchProvider",
    "SearchResult",
    "get_search_provider",
    "BrowserInterface",
    "get_browser",
]
