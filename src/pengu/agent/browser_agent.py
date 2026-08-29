"""
Browser Agent — real browser interaction via Playwright.

Extends the existing BrowserInterface with actual page interaction:
  - navigate
  - find elements by text/selector
  - click elements
  - type into input fields
  - read page content
  - search on page
  - scroll
  - wait for elements

Uses Playwright for DOM interaction (preferred over screenshots).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

from pengu.agent import ActionResult, ActionType, ActionStatus
from pengu.logging import get_logger

logger = get_logger("pengu.agent.browser")


@dataclass
class BrowserState:
    """Current state of the browser session."""
    url: str = ""
    title: str = ""
    is_ready: bool = False
    last_action: str = ""
    error: str = ""


class BrowserAgent:
    """
    Real browser interaction agent using Playwright.

    Supports:
      - Navigate to URLs
      - Find elements by text, selector, or role
      - Click, type, scroll
      - Read visible text
      - Search on Google/ChatGPT/etc
      - Wait for page loads
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._page = None
        self._state = BrowserState()
        self._available = False
        self._timeout = 10000  # 10s default

    async def ensure_browser(self) -> ActionResult:
        """Ensure the browser is running and available."""
        if self._browser and self._page:
            return ActionResult.ok("Browser is ready", action=ActionType.OPEN_URL)

        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=False,
                args=["--disable-blink-features=AutomationControlled"],
            )
            context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            self._page = await context.new_page()
            self._available = True
            self._state.is_ready = True
            logger.info("browser_launched")
            return ActionResult.ok("Browser launched", action=ActionType.OPEN_URL)
        except ImportError:
            self._available = False
            return ActionResult.fail(
                "Playwright not installed. Run: pip install playwright && playwright install chromium",
                error_code="NOT_AVAILABLE",
                action=ActionType.OPEN_URL,
            )
        except Exception as e:
            self._available = False
            return ActionResult.fail(
                f"Failed to launch browser: {e}",
                error_code="LAUNCH_FAILED",
                action=ActionType.OPEN_URL,
            )

    async def navigate(self, url: str) -> ActionResult:
        """Navigate to a URL."""
        result = await self.ensure_browser()
        if not result.success:
            return result

        try:
            if not url.startswith("http"):
                url = "https://" + url
            start = time.perf_counter()
            await self._page.goto(url, wait_until="domcontentloaded", timeout=self._timeout)
            title = await self._page.title()
            elapsed = (time.perf_counter() - start) * 1000
            self._state.url = url
            self._state.title = title
            self._state.last_action = f"navigate:{url}"
            logger.info("browser_navigated", url=url, title=title[:60], ms=round(elapsed))
            return ActionResult.ok(
                f"Opened {title}" if title else f"Opened {url}",
                action=ActionType.NAVIGATE,
                target=url,
                verified=True,
                metadata={"title": title, "url": url},
            )
        except Exception as e:
            logger.error("browser_navigate_failed", url=url, error=str(e))
            return ActionResult.fail(
                f"Could not open {url}: {e}",
                action=ActionType.NAVIGATE,
                target=url,
            )

    async def find_and_click(self, text: str) -> ActionResult:
        """Find an element by visible text and click it."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.CLICK, target=text)

        try:
            # Try multiple strategies to find the element
            strategies = [
                # Try role-based first (most reliable)
                lambda: self._page.get_by_role("link", name=text).first,
                lambda: self._page.get_by_role("button", name=text).first,
                lambda: self._page.get_by_text(text, exact=False).first,
                # Try text selector
                lambda: self._page.locator(f"text={text}").first,
            ]

            for strategy in strategies:
                try:
                    element = strategy()
                    if await element.is_visible(timeout=2000):
                        await element.click(timeout=self._timeout)
                        await self._page.wait_for_load_state("domcontentloaded", timeout=5000)
                        title = await self._page.title()
                        self._state.title = title
                        self._state.last_action = f"click:{text}"
                        logger.info("browser_clicked", text=text, title=title[:60])
                        return ActionResult.ok(
                            f"Clicked '{text}'",
                            action=ActionType.CLICK,
                            target=text,
                            verified=True,
                        )
                except Exception:
                    continue

            return ActionResult.not_found(
                f"Could not find '{text}' on the page",
                action=ActionType.CLICK,
                target=text,
            )
        except Exception as e:
            return ActionResult.fail(
                f"Error clicking '{text}': {e}",
                action=ActionType.CLICK,
                target=text,
            )

    async def search_google(self, query: str) -> ActionResult:
        """Navigate to Google and search for a query."""
        result = await self.navigate(f"https://www.google.com/search?q={query.replace(' ', '+')}")
        if result.success:
            result.message = f"Searching Google for {query}"
        return result

    async def search_chatgpt(self, query: str) -> ActionResult:
        """Navigate to ChatGPT and enter a query."""
        result = await self.navigate("https://chatgpt.com")
        if not result.success:
            return result

        try:
            # Wait for ChatGPT to load and find the input field
            await self._page.wait_for_load_state("networkidle", timeout=15000)
            await asyncio.sleep(2)

            # Try to find the text input
            input_selectors = [
                "textarea",
                "#prompt-textarea",
                "[contenteditable='true']",
                "div[role='textbox']",
            ]

            for selector in input_selectors:
                try:
                    element = self._page.locator(selector).first
                    if await element.is_visible(timeout=3000):
                        await element.click()
                        await element.fill(query)
                        await asyncio.sleep(0.5)
                        # Press Enter to submit
                        await element.press("Enter")
                        self._state.last_action = f"chatgpt_search:{query}"
                        logger.info("chatgpt_query_submitted", query=query)
                        return ActionResult.ok(
                            f"Submitted query to ChatGPT: {query}",
                            action=ActionType.SEARCH,
                            target=query,
                            verified=True,
                        )
                except Exception:
                    continue

            return ActionResult.fail(
                "Could not find ChatGPT input field",
                action=ActionType.SEARCH,
                target=query,
            )
        except Exception as e:
            return ActionResult.fail(
                f"Error with ChatGPT: {e}",
                action=ActionType.SEARCH,
                target=query,
            )

    async def type_in_page(self, text: str) -> ActionResult:
        """Type text into the currently focused element on the page."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.TYPE_TEXT)

        try:
            await self._page.keyboard.type(text, delay=30)
            return ActionResult.ok(f"Typed '{text}'", action=ActionType.TYPE_TEXT, target=text)
        except Exception as e:
            return ActionResult.fail(f"Typing error: {e}", action=ActionType.TYPE_TEXT)

    async def press_key(self, key: str) -> ActionResult:
        """Press a keyboard key in the browser."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.PRESS_KEY)

        try:
            await self._page.keyboard.press(key)
            return ActionResult.ok(f"Pressed {key}", action=ActionType.PRESS_KEY, target=key)
        except Exception as e:
            return ActionResult.fail(f"Key press error: {e}", action=ActionType.PRESS_KEY)

    async def scroll_page(self, direction: str = "down", amount: int = 3) -> ActionResult:
        """Scroll the page in a direction."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.SCROLL)

        try:
            delta = -500 if direction == "down" else 500
            for _ in range(amount):
                await self._page.mouse.wheel(0, delta)
                await asyncio.sleep(0.1)
            return ActionResult.ok(f"Scrolled {direction}", action=ActionType.SCROLL)
        except Exception as e:
            return ActionResult.fail(f"Scroll error: {e}", action=ActionType.SCROLL)

    async def read_page(self, max_chars: int = 5000) -> ActionResult:
        """Read the visible text content of the current page."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.READ_PAGE)

        try:
            # Get visible text content
            text = await self._page.inner_text("body")
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"
            title = await self._page.title()
            self._state.title = title
            return ActionResult.ok(
                text,
                action=ActionType.READ_PAGE,
                verified=True,
                metadata={"title": title, "url": self._state.url},
            )
        except Exception as e:
            return ActionResult.fail(f"Read error: {e}", action=ActionType.READ_PAGE)

    async def get_page_info(self) -> dict[str, str]:
        """Get basic page info."""
        if not self._page:
            return {"url": "", "title": "", "status": "not_open"}
        try:
            title = await self._page.title()
            return {
                "url": self._page.url,
                "title": title,
                "status": "ready",
            }
        except Exception:
            return {"url": "", "title": "", "status": "error"}

    async def close(self) -> None:
        """Close the browser."""
        try:
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._page = None
        self._playwright = None
        self._state = BrowserState()
        logger.info("browser_closed")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def state(self) -> BrowserState:
        return self._state


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_agent: Optional[BrowserAgent] = None


def get_browser_agent() -> BrowserAgent:
    """Get the global browser agent."""
    global _agent
    if _agent is None:
        _agent = BrowserAgent()
    return _agent
