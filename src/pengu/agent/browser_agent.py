"""
Browser Agent — real browser interaction via Playwright.

Extends the existing BrowserInterface with actual page interaction:
  - navigate
  - structured page observation (BrowserState)
  - interactive element discovery
  - find elements by text/selector/role
  - click elements
  - type into specific input fields
  - type into the focused element
  - submit forms
  - read page content
  - search on page
  - scroll
  - wait for navigation/elements
  - verify actions succeeded

Uses Playwright for DOM interaction (preferred over screenshots).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pengu.agent import ActionResult, ActionType, ActionStatus
from pengu.logging import get_logger

logger = get_logger("pengu.agent.browser")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class InteractiveElement:
    """A discovered interactive element on the page."""
    tag: str = ""
    element_type: str = ""   # "input", "button", "link", "textarea", "select", "other"
    text: str = ""            # visible text / label
    placeholder: str = ""
    aria_label: str = ""
    name: str = ""            # name attribute
    href: str = ""            # for links
    role: str = ""            # ARIA role
    selector: str = ""        # best-guess CSS/Playwright selector
    is_visible: bool = True
    is_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "type": self.element_type,
            "text": self.text[:80],
            "placeholder": self.placeholder,
            "aria_label": self.aria_label,
            "role": self.role,
            "selector": self.selector,
            "visible": self.is_visible,
            "enabled": self.is_enabled,
        }


@dataclass
class BrowserState:
    """Complete structured state of the browser session."""
    url: str = ""
    title: str = ""
    is_ready: bool = False
    loading: bool = False
    error: str = ""

    # Page structure
    interactive_elements: list[InteractiveElement] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)
    headings: list[str] = field(default_factory=list)
    visible_text_preview: str = ""

    # Navigation state
    can_go_back: bool = False
    can_go_forward: bool = False

    last_action: str = ""
    last_action_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "ready": self.is_ready,
            "loading": self.loading,
            "error": self.error,
            "interactive_count": len(self.interactive_elements),
            "interactive_elements": [e.to_dict() for e in self.interactive_elements[:20]],
            "links_count": len(self.links),
            "headings": self.headings[:10],
            "text_preview": self.visible_text_preview[:500],
            "can_go_back": self.can_go_back,
            "can_go_forward": self.can_go_forward,
            "last_action": self.last_action,
        }

    def get_summary(self) -> str:
        """Human-readable summary for the agent brain."""
        parts = [f"Page: {self.title}" if self.title else "Page: (no title)"]
        parts.append(f"URL: {self.url}")
        if self.error:
            parts.append(f"Error: {self.error}")

        inputs = [e for e in self.interactive_elements if e.element_type in ("input", "textarea")]
        buttons = [e for e in self.interactive_elements if e.element_type == "button"]
        links = [e for e in self.interactive_elements if e.element_type == "link"]

        if inputs:
            labels = [e.placeholder or e.aria_label or e.text or e.name for e in inputs[:5]]
            parts.append(f"Input fields: {', '.join(labels)}")
        if buttons:
            labels = [e.text or e.aria_label for e in buttons[:5]]
            parts.append(f"Buttons: {', '.join(labels)}")
        if links:
            labels = [e.text[:30] for e in links[:5]]
            parts.append(f"Links: {', '.join(labels)}")

        return " | ".join(parts)


class BrowserAgent:
    """
    Real browser interaction agent using Playwright.

    Supports:
      - Navigate to URLs
      - Structured page observation (BrowserState)
      - Interactive element discovery (semantic, not just CSS selectors)
      - Find elements by text, selector, or role
      - Click, type, scroll
      - Type into specific input fields by label/placeholder
      - Submit forms
      - Read visible text
      - Search on Google/ChatGPT/etc
      - Wait for page loads and navigation
      - Verify actions succeeded
    """

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._state = BrowserState()
        self._available = False
        self._timeout = 15000  # 15s default
        self._navigation_history: list[str] = []

    async def ensure_browser(self) -> ActionResult:
        """Ensure the browser is running and available."""
        if self._browser and self._page:
            return ActionResult.ok("Browser is ready", action=ActionType.OPEN_URL)

        try:
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=False,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--start-maximized",
                ],
            )
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/131.0.0.0 Safari/537.36"
                ),
            )
            self._page = await self._context.new_page()
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

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

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

            self._navigation_history.append(url)
            self._state.url = url
            self._state.title = title
            self._state.last_action = f"navigate:{url}"
            self._state.last_action_time = time.time()
            self._state.error = ""

            logger.info("browser_navigated", url=url, title=title[:60], ms=round(elapsed))
            return ActionResult.ok(
                f"Opened {title}" if title else f"Opened {url}",
                action=ActionType.NAVIGATE,
                target=url,
                verified=True,
                metadata={"title": title, "url": url, "duration_ms": round(elapsed)},
            )
        except Exception as e:
            logger.error("browser_navigate_failed", url=url, error=str(e))
            self._state.error = str(e)
            return ActionResult.fail(
                f"Could not open {url}: {e}",
                action=ActionType.NAVIGATE,
                target=url,
            )

    async def go_back(self) -> ActionResult:
        """Navigate back in browser history."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.NAVIGATE)
        try:
            await self._page.go_back(wait_until="domcontentloaded", timeout=self._timeout)
            title = await self._page.title()
            self._state.url = self._page.url
            self._state.title = title
            self._state.last_action = "go_back"
            return ActionResult.ok(f"Back to: {title}", action=ActionType.NAVIGATE, verified=True)
        except Exception as e:
            return ActionResult.fail(f"Go back failed: {e}", action=ActionType.NAVIGATE)

    async def go_forward(self) -> ActionResult:
        """Navigate forward in browser history."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.NAVIGATE)
        try:
            await self._page.go_forward(wait_until="domcontentloaded", timeout=self._timeout)
            title = await self._page.title()
            self._state.url = self._page.url
            self._state.title = title
            self._state.last_action = "go_forward"
            return ActionResult.ok(f"Forward to: {title}", action=ActionType.NAVIGATE, verified=True)
        except Exception as e:
            return ActionResult.fail(f"Go forward failed: {e}", action=ActionType.NAVIGATE)

    async def refresh(self) -> ActionResult:
        """Refresh the current page."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.NAVIGATE)
        try:
            await self._page.reload(wait_until="domcontentloaded", timeout=self._timeout)
            title = await self._page.title()
            self._state.title = title
            self._state.last_action = "refresh"
            return ActionResult.ok(f"Refreshed: {title}", action=ActionType.NAVIGATE, verified=True)
        except Exception as e:
            return ActionResult.fail(f"Refresh failed: {e}", action=ActionType.NAVIGATE)

    # ------------------------------------------------------------------
    # Structured observation
    # ------------------------------------------------------------------

    async def get_browser_state(self) -> BrowserState:
        """
        Observe the current browser page and return a structured BrowserState.

        This is the primary observation method the agent uses before acting.
        It discovers interactive elements semantically (roles, labels, text)
        rather than relying on fragile CSS selectors.
        """
        if not self._page:
            self._state.error = "Browser not open"
            return self._state

        try:
            self._state.loading = True

            # Basic page info
            self._state.url = self._page.url
            self._state.title = await self._page.title()
            self._state.error = ""

            # Discover interactive elements
            self._state.interactive_elements = await self._discover_interactive_elements()

            # Discover links
            self._state.links = await self._discover_links()

            # Discover headings
            self._state.headings = await self._discover_headings()

            # Get text preview
            self._state.visible_text_preview = await self._get_visible_text_preview(max_chars=2000)

            self._state.is_ready = True
            self._state.loading = False
            self._state.last_action = "observe"
            self._state.last_action_time = time.time()

            logger.info(
                "browser_state_observed",
                url=self._state.url[:80],
                title=self._state.title[:60],
                elements=len(self._state.interactive_elements),
                links=len(self._state.links),
            )
            return self._state

        except Exception as e:
            self._state.loading = False
            self._state.error = str(e)
            logger.error("browser_state_failed", error=str(e))
            return self._state

    async def _discover_interactive_elements(self) -> list[InteractiveElement]:
        """Discover all interactive elements on the page using semantic methods."""
        elements: list[InteractiveElement] = []
        seen = set()  # deduplicate

        try:
            # Strategy 1: Input fields (text, search, email, password, etc.)
            inputs = await self._page.query_selector_all(
                "input:visible, textarea:visible, select:visible"
            )
            for el in inputs:
                info = await self._extract_input_info(el)
                if info and info.text not in seen:
                    seen.add(info.text)
                    elements.append(info)

            # Strategy 2: Buttons
            buttons = await self._page.query_selector_all(
                "button:visible, [role='button']:visible, input[type='submit']:visible"
            )
            for el in buttons:
                info = await self._extract_button_info(el)
                if info and info.text not in seen:
                    seen.add(info.text)
                    elements.append(info)

            # Strategy 3: Links with meaningful text
            links = await self._page.query_selector_all("a:visible")
            for el in links:
                info = await self._extract_link_info(el)
                if info and info.text and info.text not in seen and len(info.text) > 1:
                    seen.add(info.text)
                    elements.append(info)

        except Exception as e:
            logger.debug("element_discovery_partial", error=str(e))

        return elements

    async def _extract_input_info(self, el) -> Optional[InteractiveElement]:
        """Extract info from an input/textarea/select element."""
        try:
            tag = await el.evaluate("e => e.tagName.toLowerCase()")
            input_type = await el.get_attribute("type") or ""
            placeholder = (await el.get_attribute("placeholder")) or ""
            aria_label = (await el.get_attribute("aria-label")) or ""
            name = (await el.get_attribute("name")) or ""
            value = (await el.get_attribute("value")) or ""

            # Build a selector for later use
            selector = await self._build_selector(el)

            # The "text" for this element is its most identifiable label
            text = placeholder or aria_label or name or value or f"{tag}[{input_type}]"

            element_type = "textarea" if tag == "textarea" else "input"
            if tag == "select":
                element_type = "select"

            return InteractiveElement(
                tag=tag,
                element_type=element_type,
                text=text,
                placeholder=placeholder,
                aria_label=aria_label,
                name=name,
                selector=selector,
                is_visible=True,
                is_enabled=await el.is_enabled(),
            )
        except Exception:
            return None

    async def _extract_button_info(self, el) -> Optional[InteractiveElement]:
        """Extract info from a button element."""
        try:
            tag = await el.evaluate("e => e.tagName.toLowerCase()")
            text = (await el.inner_text()).strip()
            aria_label = (await el.get_attribute("aria-label")) or ""
            role = (await el.get_attribute("role")) or ""
            selector = await self._build_selector(el)

            display_text = text or aria_label or f"[{tag}]"
            return InteractiveElement(
                tag=tag,
                element_type="button",
                text=display_text,
                aria_label=aria_label,
                role=role,
                selector=selector,
                is_visible=True,
                is_enabled=await el.is_enabled(),
            )
        except Exception:
            return None

    async def _extract_link_info(self, el) -> Optional[InteractiveElement]:
        """Extract info from a link element."""
        try:
            text = (await el.inner_text()).strip()
            href = (await el.get_attribute("href")) or ""
            aria_label = (await el.get_attribute("aria-label")) or ""
            selector = await self._build_selector(el)

            return InteractiveElement(
                tag="a",
                element_type="link",
                text=text[:100],
                href=href,
                aria_label=aria_label,
                selector=selector,
                is_visible=True,
            )
        except Exception:
            return None

    async def _build_selector(self, el) -> str:
        """Build the best available selector for an element."""
        try:
            # Try id first
            el_id = await el.get_attribute("id")
            if el_id:
                return f"#{el_id}"

            # Try data-testid
            test_id = await el.get_attribute("data-testid")
            if test_id:
                return f"[data-testid='{test_id}']"

            # Try name attribute
            name = await el.get_attribute("name")
            if name:
                tag = await el.evaluate("e => e.tagName.toLowerCase()")
                return f"{tag}[name='{name}']"

            # Try aria-label
            aria = await el.get_attribute("aria-label")
            if aria:
                tag = await el.evaluate("e => e.tagName.toLowerCase()")
                role = await el.get_attribute("role") or ""
                if role:
                    return f"[role='{role}'][aria-label='{aria}']"
                return f"{tag}[aria-label='{aria}']"

            # Fallback: use text content
            text = (await el.inner_text()).strip()[:30]
            if text:
                return f"text={text}"

            return ""
        except Exception:
            return ""

    async def _discover_links(self) -> list[dict[str, str]]:
        """Discover links on the page."""
        links = []
        try:
            elements = await self._page.query_selector_all("a[href]:visible")
            for el in elements[:50]:
                text = (await el.inner_text()).strip()
                href = (await el.get_attribute("href")) or ""
                if text and href and len(text) > 1:
                    links.append({"text": text[:100], "href": href[:200]})
        except Exception:
            pass
        return links

    async def _discover_headings(self) -> list[str]:
        """Discover page headings."""
        headings = []
        try:
            elements = await self._page.query_selector_all("h1, h2, h3")
            for el in elements[:15]:
                text = (await el.inner_text()).strip()
                if text:
                    headings.append(text[:100])
        except Exception:
            pass
        return headings

    async def _get_visible_text_preview(self, max_chars: int = 2000) -> str:
        """Get a preview of visible text content."""
        try:
            text = await self._page.inner_text("body")
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            cleaned = "\n".join(lines)
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars] + "\n... (truncated)"
            return cleaned
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Element interaction — semantic (find by text/role/label)
    # ------------------------------------------------------------------

    async def find_and_click(self, text: str) -> ActionResult:
        """Find an element by visible text/label and click it."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.CLICK, target=text)

        try:
            # Multi-strategy element discovery
            strategies = [
                # ARIA role-based (most reliable)
                ("role_link", lambda: self._page.get_by_role("link", name=text).first),
                ("role_button", lambda: self._page.get_by_role("button", name=text).first),
                ("role_tab", lambda: self._page.get_by_role("tab", name=text).first),
                ("role_menuitem", lambda: self._page.get_by_role("menuitem", name=text).first),
                # Label-based
                ("label", lambda: self._page.get_by_label(text).first),
                # Text-based (partial match)
                ("text_exact", lambda: self._page.get_by_text(text, exact=True).first),
                ("text_partial", lambda: self._page.get_by_text(text, exact=False).first),
                # Placeholder-based
                ("placeholder", lambda: self._page.get_by_placeholder(text).first),
                # CSS selector fallback
                ("css_text", lambda: self._page.locator(f"text={text}").first),
            ]

            for strategy_name, strategy_fn in strategies:
                try:
                    element = strategy_fn()
                    if await element.is_visible(timeout=2000):
                        await element.click(timeout=self._timeout)
                        # Wait for navigation or DOM update
                        try:
                            await self._page.wait_for_load_state(
                                "domcontentloaded", timeout=5000
                            )
                        except Exception:
                            pass
                        title = await self._page.title()
                        self._state.title = title
                        self._state.url = self._page.url
                        self._state.last_action = f"click:{text}"
                        self._state.last_action_time = time.time()
                        logger.info("browser_clicked", text=text, strategy=strategy_name, title=title[:60])
                        return ActionResult.ok(
                            f"Clicked '{text}'",
                            action=ActionType.CLICK,
                            target=text,
                            verified=True,
                            metadata={"strategy": strategy_name, "title": title, "url": self._page.url},
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

    async def type_in_field(self, field_text: str, value: str) -> ActionResult:
        """
        Type text into a specific input field identified by its label, placeholder, or name.

        This is the primary way to interact with form fields.
        It finds the field by semantic identification, not fixed selectors.
        """
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.TYPE_TEXT)

        try:
            # Multi-strategy field discovery
            strategies = [
                ("label", lambda: self._page.get_by_label(field_text).first),
                ("placeholder", lambda: self._page.get_by_placeholder(field_text).first),
                ("role_textbox_labeled", lambda: self._page.get_by_role("textbox", name=field_text).first),
                ("role_textbox", lambda: self._page.get_by_role("textbox").first),
                ("aria_label", lambda: self._page.locator(f"[aria-label*='{field_text}']").first),
                ("name", lambda: self._page.locator(f"[name*='{field_text}' i]").first),
                ("css_input", lambda: self._page.locator(f"input[placeholder*='{field_text}' i]").first),
                ("css_textarea", lambda: self._page.locator(f"textarea[placeholder*='{field_text}' i]").first),
            ]

            for strategy_name, strategy_fn in strategies:
                try:
                    element = strategy_fn()
                    if await element.is_visible(timeout=2000):
                        # Clear existing text and type new value
                        await element.click()
                        await element.fill("")
                        await element.fill(value)
                        await asyncio.sleep(0.3)
                        self._state.last_action = f"type_in:{field_text}={value[:30]}"
                        self._state.last_action_time = time.time()
                        logger.info(
                            "browser_typed_in_field",
                            field=field_text,
                            value=value[:30],
                            strategy=strategy_name,
                        )
                        return ActionResult.ok(
                            f"Typed '{value[:50]}' into '{field_text}'",
                            action=ActionType.TYPE_TEXT,
                            target=field_text,
                            verified=True,
                            metadata={"strategy": strategy_name, "value": value},
                        )
                except Exception:
                    continue

            # If no specific field found, try typing into any visible textbox
            try:
                textboxes = self._page.locator("input[type='text']:visible, textarea:visible, [contenteditable='true']:visible")
                count = await textboxes.count()
                if count > 0:
                    first = textboxes.first
                    await first.click()
                    await first.fill("")
                    await first.fill(value)
                    self._state.last_action = f"type_in_any={value[:30]}"
                    return ActionResult.ok(
                        f"Typed '{value[:50]}' into the first text field",
                        action=ActionType.TYPE_TEXT,
                        target=field_text,
                        verified=True,
                    )
            except Exception:
                pass

            return ActionResult.not_found(
                f"Could not find input field '{field_text}'",
                action=ActionType.TYPE_TEXT,
                target=field_text,
            )
        except Exception as e:
            return ActionResult.fail(f"Error typing into field: {e}", action=ActionType.TYPE_TEXT)

    async def type_in_page(self, text: str) -> ActionResult:
        """Type text into the currently focused element on the page."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.TYPE_TEXT)

        try:
            await self._page.keyboard.type(text, delay=30)
            self._state.last_action = f"type:{text[:30]}"
            self._state.last_action_time = time.time()
            return ActionResult.ok(f"Typed '{text[:50]}'", action=ActionType.TYPE_TEXT, target=text)
        except Exception as e:
            return ActionResult.fail(f"Typing error: {e}", action=ActionType.TYPE_TEXT)

    async def press_key(self, key: str) -> ActionResult:
        """Press a keyboard key in the browser."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.PRESS_KEY)

        try:
            await self._page.keyboard.press(key)
            self._state.last_action = f"press:{key}"
            return ActionResult.ok(f"Pressed {key}", action=ActionType.PRESS_KEY, target=key)
        except Exception as e:
            return ActionResult.fail(f"Key press error: {e}", action=ActionType.PRESS_KEY)

    async def submit_form(self) -> ActionResult:
        """Submit the current form (press Enter on the active element)."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.PRESS_KEY)

        try:
            await self._page.keyboard.press("Enter")
            # Wait for navigation or network idle
            try:
                await self._page.wait_for_load_state("domcontentloaded", timeout=8000)
            except Exception:
                pass
            title = await self._page.title()
            self._state.title = title
            self._state.url = self._page.url
            self._state.last_action = "submit_form"
            self._state.last_action_time = time.time()
            logger.info("browser_form_submitted", title=title[:60])
            return ActionResult.ok(
                f"Form submitted. Page: {title}",
                action=ActionType.PRESS_KEY,
                target="Enter",
                verified=True,
                metadata={"title": title, "url": self._page.url},
            )
        except Exception as e:
            return ActionResult.fail(f"Submit failed: {e}", action=ActionType.PRESS_KEY)

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

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
            await self._page.wait_for_load_state("networkidle", timeout=15000)
            await asyncio.sleep(2)

            # Find the ChatGPT input field
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
            return ActionResult.fail(f"Error with ChatGPT: {e}", action=ActionType.SEARCH)

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    async def scroll_page(self, direction: str = "down", amount: int = 3) -> ActionResult:
        """Scroll the page in a direction."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.SCROLL)

        try:
            delta = -500 if direction == "down" else 500
            for _ in range(amount):
                await self._page.mouse.wheel(0, delta)
                await asyncio.sleep(0.1)
            self._state.last_action = f"scroll:{direction}"
            return ActionResult.ok(f"Scrolled {direction}", action=ActionType.SCROLL)
        except Exception as e:
            return ActionResult.fail(f"Scroll error: {e}", action=ActionType.SCROLL)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    async def read_page(self, max_chars: int = 5000) -> ActionResult:
        """Read the visible text content of the current page."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.READ_PAGE)

        try:
            text = await self._page.inner_text("body")
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... (truncated)"
            title = await self._page.title()
            self._state.title = title
            self._state.last_action = "read_page"
            return ActionResult.ok(
                text,
                action=ActionType.READ_PAGE,
                verified=True,
                metadata={"title": title, "url": self._state.url, "length": len(text)},
            )
        except Exception as e:
            return ActionResult.fail(f"Read error: {e}", action=ActionType.READ_PAGE)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    async def verify_action(
        self,
        expected_url: Optional[str] = None,
        expected_title: Optional[str] = None,
        expected_text: Optional[str] = None,
        timeout_ms: int = 5000,
    ) -> ActionResult:
        """
        Verify that the current page state matches expectations.

        Checks URL, title, and/or visible text.
        """
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.VERIFY)

        try:
            await asyncio.sleep(0.5)  # brief wait for state to settle

            actual_url = self._page.url
            actual_title = await self._page.title()

            checks = []
            passed = True

            if expected_url:
                if expected_url.lower() in actual_url.lower():
                    checks.append(f"URL matches: {actual_url}")
                else:
                    checks.append(f"URL mismatch: expected '{expected_url}', got '{actual_url}'")
                    passed = False

            if expected_title:
                if expected_title.lower() in actual_title.lower():
                    checks.append(f"Title matches: {actual_title}")
                else:
                    checks.append(f"Title mismatch: expected '{expected_title}', got '{actual_title}'")
                    passed = False

            if expected_text:
                body_text = await self._page.inner_text("body")
                if expected_text.lower() in body_text.lower():
                    checks.append(f"Text found: '{expected_text}'")
                else:
                    checks.append(f"Text not found: '{expected_text}'")
                    passed = False

            message = "; ".join(checks) if checks else "No checks specified"
            self._state.last_action = "verify"

            if passed:
                return ActionResult.ok(
                    f"Verification passed: {message}",
                    action=ActionType.VERIFY,
                    verified=True,
                    metadata={"url": actual_url, "title": actual_title},
                )
            else:
                return ActionResult.fail(
                    f"Verification failed: {message}",
                    action=ActionType.VERIFY,
                    metadata={"url": actual_url, "title": actual_title},
                )
        except Exception as e:
            return ActionResult.fail(f"Verification error: {e}", action=ActionType.VERIFY)

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------

    async def wait_for_element(self, selector: str, timeout_ms: int = 5000) -> ActionResult:
        """Wait for an element to appear on the page."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.WAIT)

        try:
            await self._page.wait_for_selector(selector, timeout=timeout_ms)
            return ActionResult.ok(
                f"Element found: {selector}",
                action=ActionType.WAIT,
                verified=True,
            )
        except Exception:
            return ActionResult.fail(
                f"Element not found within {timeout_ms}ms: {selector}",
                action=ActionType.WAIT,
            )

    async def wait_for_navigation(self, timeout_ms: int = 10000) -> ActionResult:
        """Wait for navigation to complete."""
        if not self._page:
            return ActionResult.fail("Browser not open", action=ActionType.WAIT)

        try:
            await self._page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            title = await self._page.title()
            self._state.title = title
            self._state.url = self._page.url
            return ActionResult.ok(
                f"Navigation complete: {title}",
                action=ActionType.WAIT,
                verified=True,
                metadata={"title": title, "url": self._page.url},
            )
        except Exception:
            return ActionResult.fail(
                "Navigation timeout",
                action=ActionType.WAIT,
            )

    # ------------------------------------------------------------------
    # Page info
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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
        self._context = None
        self._page = None
        self._playwright = None
        self._state = BrowserState()
        self._navigation_history.clear()
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
