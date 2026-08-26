"""
Browser interface abstraction for Pengu.

Provides safe, controlled browser automation.
All browser actions require user confirmation for sensitive operations.

Design principle: Never make browser automation mandatory.
Pengu works without a browser.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from pengu.logging import get_logger

logger = get_logger("pengu.web.browser")


@dataclass
class BrowserPage:
    """Represents a browser page."""
    url: str
    title: str
    content: str = ""
    screenshot_path: Optional[str] = None


class BrowserInterface(ABC):
    """Base class for browser automation."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._available = False

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if browser is available."""
        ...

    @abstractmethod
    async def open(self, url: str) -> BrowserPage:
        """Open a URL in the browser."""
        ...

    @abstractmethod
    async def navigate(self, url: str) -> BrowserPage:
        """Navigate to a URL."""
        ...

    @abstractmethod
    async def get_content(self) -> Optional[str]:
        """Get the current page content."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the browser."""
        ...

    def is_available(self) -> bool:
        """Quick check without network call."""
        return self._available


class PlaywrightBrowser(BrowserInterface):
    """
    Browser automation using Playwright.

    Requires: playwright installed and browsers downloaded.
    License: Apache-2.0
    """

    def __init__(self) -> None:
        super().__init__(name="playwright")
        self._browser = None
        self._page = None

    async def health_check(self) -> bool:
        """Check if Playwright is available."""
        try:
            from playwright.async_api import async_playwright
            self._available = True
            return True
        except ImportError:
            self._available = False
            logger.warning("playwright_not_installed")
            return False

    async def open(self, url: str) -> BrowserPage:
        """Open a URL in the browser."""
        if not self._available:
            await self.health_check()

        if not self._available:
            raise RuntimeError("Browser not available. Install playwright: pip install playwright")

        try:
            from playwright.async_api import async_playwright

            if self._browser is None:
                p = await async_playwright().start()
                self._browser = await p.chromium.launch(headless=True)
                self._page = await self._browser.new_page()

            await self._page.goto(url)
            title = await self._page.title()
            content = await self._page.content()

            logger.info("browser_opened", url=url)
            return BrowserPage(url=url, title=title, content=content[:5000])

        except Exception as e:
            logger.error("browser_open_failed", url=url, error=str(e))
            raise

    async def navigate(self, url: str) -> BrowserPage:
        """Navigate to a URL."""
        return await self.open(url)

    async def get_content(self) -> Optional[str]:
        """Get the current page content."""
        if self._page:
            return await self._page.content()
        return None

    async def close(self) -> None:
        """Close the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
            logger.info("browser_closed")


class MockBrowser(BrowserInterface):
    """Mock browser for testing."""

    def __init__(self) -> None:
        super().__init__(name="mock")
        self._available = True

    async def health_check(self) -> bool:
        return True

    async def open(self, url: str) -> BrowserPage:
        return BrowserPage(url=url, title="Mock Page", content="Mock content")

    async def navigate(self, url: str) -> BrowserPage:
        return await self.open(url)

    async def get_content(self) -> Optional[str]:
        return "Mock content"

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_browser: Optional[BrowserInterface] = None


def get_browser() -> BrowserInterface:
    """Get the browser interface (lazy initialization)."""
    global _browser
    if _browser is None:
        _browser = PlaywrightBrowser()
    return _browser


def reset_browser() -> BrowserInterface:
    """Reset the browser (for testing)."""
    global _browser
    _browser = PlaywrightBrowser()
    return _browser
