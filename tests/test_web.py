"""
Tests for Pengu web search and browser interfaces.
"""
import pytest
from pengu.web.search import (
    DuckDuckGoProvider,
    MockSearchProvider,
    SearchResult,
    WebSearchProvider,
)


class TestSearchResult:
    def test_to_dict(self):
        r = SearchResult(title="Test", url="http://example.com", snippet="A snippet", rank=1)
        d = r.to_dict()
        assert d["title"] == "Test"
        assert d["url"] == "http://example.com"
        assert d["snippet"] == "A snippet"
        assert d["rank"] == 1


class TestMockSearchProvider:
    async def test_health_check(self):
        provider = MockSearchProvider()
        assert await provider.health_check() is True

    async def test_search_returns_results(self):
        provider = MockSearchProvider()
        provider.set_results([
            SearchResult(title="Result 1", url="http://1.com", snippet="Snippet 1"),
            SearchResult(title="Result 2", url="http://2.com", snippet="Snippet 2"),
        ])
        results = await provider.search("test")
        assert len(results) == 2
        assert results[0].title == "Result 1"

    async def test_fetch(self):
        provider = MockSearchProvider()
        content = await provider.fetch("http://example.com")
        assert content is not None
        assert "Mock content" in content


class TestDuckDuckGoProvider:
    def test_init(self):
        provider = DuckDuckGoProvider()
        assert provider.name == "duckduckgo"

    async def test_health_check_works(self):
        provider = DuckDuckGoProvider()
        result = await provider.health_check()
        assert isinstance(result, bool)


class TestBrowserInterface:
    async def test_mock_browser(self):
        from pengu.web.browser import MockBrowser
        browser = MockBrowser()
        assert await browser.health_check() is True

        page = await browser.open("http://example.com")
        assert page.title == "Mock Page"
        assert page.url == "http://example.com"

    def test_playwright_browser_init(self):
        from pengu.web.browser import PlaywrightBrowser
        browser = PlaywrightBrowser()
        assert browser.name == "playwright"
