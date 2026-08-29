"""
Agent Tools — register desktop, browser, screen, and web search tools
into the unified ToolRegistry.

This module bridges the existing agent subsystems (DesktopAutomation,
BrowserAgent, ScreenObserver, DuckDuckGoProvider) into the Tool interface
that AgentBrain and MissionManager use.

Every tool follows the same schema:
  name, description, category, permission_level, parameters, handler

The AgentBrain selects tools by name. It does NOT depend on individual
implementation details.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any

from pengu.config import PermissionLevel
from pengu.logging import get_logger
from pengu.tools.registry import Tool, ToolRegistry, ToolResult

logger = get_logger("pengu.agent.tools")


# ---------------------------------------------------------------------------
# Helpers — run async code from sync context safely
# ---------------------------------------------------------------------------

def _run_async(coro) -> Any:
    """Run an async coroutine from a sync context, handling event-loop conflicts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside a running loop — run in a thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            def _run():
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            future = pool.submit(_run)
            return future.result(timeout=30)
    elif loop:
        return loop.run_until_complete(coro)
    else:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()


# ---------------------------------------------------------------------------
# Browser tools
# ---------------------------------------------------------------------------

async def browser_open(url: str) -> ToolResult:
    """Open a URL in the browser."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    result = await agent.navigate(url)
    return ToolResult(
        success=result.success,
        output={"message": result.message, "url": url},
        error=result.error,
    )


async def browser_navigate(url: str) -> ToolResult:
    """Navigate to a URL in the browser."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    result = await agent.navigate(url)
    return ToolResult(
        success=result.success,
        output={"message": result.message, "url": url},
        error=result.error,
    )


async def browser_back() -> ToolResult:
    """Navigate back in browser history."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    if not agent._page:
        return ToolResult(success=False, error="Browser not open")
    try:
        await agent._page.go_back()
        title = await agent._page.title()
        return ToolResult(success=True, output={"message": f"Navigated back. Page: {title}"})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def browser_forward() -> ToolResult:
    """Navigate forward in browser history."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    if not agent._page:
        return ToolResult(success=False, error="Browser not open")
    try:
        await agent._page.go_forward()
        title = await agent._page.title()
        return ToolResult(success=True, output={"message": f"Navigated forward. Page: {title}"})
    except Exception as e:
        return ToolResult(success=False, error=str(e))


async def browser_click(text: str) -> ToolResult:
    """Click an element by visible text on the current page."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    result = await agent.find_and_click(text)
    return ToolResult(
        success=result.success,
        output={"message": result.message},
        error=result.error,
    )


async def browser_type(text: str) -> ToolResult:
    """Type text into the currently focused element on the page."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    result = await agent.type_in_page(text)
    return ToolResult(
        success=result.success,
        output={"message": result.message},
        error=result.error,
    )


async def browser_scroll(direction: str = "down", amount: int = 3) -> ToolResult:
    """Scroll the page up or down."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    result = await agent.scroll_page(direction=direction, amount=amount)
    return ToolResult(
        success=result.success,
        output={"message": result.message},
        error=result.error,
    )


async def browser_read(max_chars: int = 5000) -> ToolResult:
    """Read the visible text content of the current page."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    result = await agent.read_page(max_chars=max_chars)
    return ToolResult(
        success=result.success,
        output={"text": result.message if result.success else "", "metadata": result.metadata},
        error=result.error,
    )


async def browser_search(query: str) -> ToolResult:
    """Search Google for a query via the browser."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    result = await agent.search_google(query)
    return ToolResult(
        success=result.success,
        output={"message": result.message},
        error=result.error,
    )


async def browser_get_url() -> ToolResult:
    """Get the current browser URL."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    info = await agent.get_page_info()
    return ToolResult(
        success=info.get("status") == "ready",
        output=info,
    )


async def browser_get_title() -> ToolResult:
    """Get the current browser page title."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    info = await agent.get_page_info()
    return ToolResult(
        success=info.get("status") == "ready",
        output={"title": info.get("title", ""), "url": info.get("url", "")},
    )


async def browser_close() -> ToolResult:
    """Close the browser."""
    from pengu.agent.browser_agent import get_browser_agent
    agent = get_browser_agent()
    await agent.close()
    return ToolResult(success=True, output={"message": "Browser closed."})


# ---------------------------------------------------------------------------
# Desktop tools
# ---------------------------------------------------------------------------

def desktop_click(x: int, y: int) -> ToolResult:
    """Click at screen coordinates."""
    from pengu.agent.desktop import get_desktop
    desktop = get_desktop()
    desktop.mouse.click(x, y)
    return ToolResult(success=True, output={"message": f"Clicked at ({x}, {y})"})


def desktop_double_click(x: int, y: int) -> ToolResult:
    """Double-click at screen coordinates."""
    from pengu.agent.desktop import get_desktop
    desktop = get_desktop()
    desktop.mouse.double_click(x, y)
    return ToolResult(success=True, output={"message": f"Double-clicked at ({x}, {y})"})


def desktop_right_click(x: int, y: int) -> ToolResult:
    """Right-click at screen coordinates."""
    from pengu.agent.desktop import get_desktop
    desktop = get_desktop()
    desktop.mouse.right_click(x, y)
    return ToolResult(success=True, output={"message": f"Right-clicked at ({x}, {y})"})


def desktop_type(text: str) -> ToolResult:
    """Type text using keyboard emulation."""
    from pengu.agent.desktop import get_desktop
    desktop = get_desktop()
    desktop.keyboard.type_text(text)
    return ToolResult(success=True, output={"message": f"Typed {len(text)} characters"})


def desktop_press(key: str) -> ToolResult:
    """Press a single key (enter, escape, tab, backspace, etc.)."""
    from pengu.agent.desktop import get_desktop
    desktop = get_desktop()
    desktop.keyboard.press_key(key)
    return ToolResult(success=True, output={"message": f"Pressed {key}"})


def desktop_hotkey(*keys: str) -> ToolResult:
    """Press a keyboard shortcut (e.g., ctrl+c, alt+tab)."""
    from pengu.agent.desktop import get_desktop
    desktop = get_desktop()
    desktop.keyboard.hotkey(*keys)
    return ToolResult(success=True, output={"message": f"Pressed {'+'.join(keys)}"})


def desktop_scroll(x: int, y: int, delta: int = -3) -> ToolResult:
    """Scroll at screen coordinates."""
    from pengu.agent.desktop import get_desktop
    desktop = get_desktop()
    desktop.mouse.scroll(x, y, delta)
    return ToolResult(success=True, output={"message": f"Scrolled at ({x}, {y})"})


def desktop_focus_window(title: str) -> ToolResult:
    """Find and focus a window by title substring."""
    from pengu.agent.desktop import get_desktop
    desktop = get_desktop()
    hwnd = desktop.window.find_window(title)
    if hwnd is None:
        return ToolResult(success=False, error=f"Window not found: {title}")
    desktop.window.restore_window(hwnd)
    desktop.window.focus_window(hwnd)
    return ToolResult(success=True, output={"message": f"Focused window: {title}"})


def desktop_open_app(application: str) -> ToolResult:
    """Open an application using the AppManager."""
    from pengu.os.app_launcher import get_launcher
    launcher = get_launcher()
    result = launcher.open_application(application)
    return ToolResult(
        success=result["success"],
        output={"message": result["message"]},
        error=result.get("error", ""),
    )


# ---------------------------------------------------------------------------
# Screen tools
# ---------------------------------------------------------------------------

def screen_get_active_window() -> ToolResult:
    """Get information about the currently active window."""
    from pengu.agent.observer import get_observer
    observer = get_observer()
    info = observer.get_active_window()
    return ToolResult(success=True, output=info)


def screen_inspect() -> ToolResult:
    """Get a summary of the current screen state."""
    from pengu.agent.observer import get_observer
    observer = get_observer()
    state = observer.get_state()
    return ToolResult(success=True, output={
        "active_window": state.active_window_title,
        "active_app": state.active_window_app,
        "screen_size": f"{state.screen_width}x{state.screen_height}",
        "elements": [e.to_dict() for e in state.elements[:15]],
    })


def screen_get_ui_tree() -> ToolResult:
    """Get UI elements of the active window."""
    from pengu.agent.observer import get_observer
    observer = get_observer()
    elements = observer.get_elements()
    return ToolResult(
        success=True,
        output={"elements": [e.to_dict() for e in elements], "count": len(elements)},
    )


# ---------------------------------------------------------------------------
# Web search tools
# ---------------------------------------------------------------------------

async def web_search(query: str, max_results: int = 5) -> ToolResult:
    """Search the web using DuckDuckGo (free, no API key)."""
    from pengu.web.search import get_search_provider
    provider = get_search_provider()
    results = await provider.search(query, max_results=max_results)
    return ToolResult(
        success=True,
        output={
            "query": query,
            "results": [r.to_dict() for r in results],
            "count": len(results),
        },
    )


async def web_fetch(url: str, max_chars: int = 10000) -> ToolResult:
    """Fetch content from a URL."""
    from pengu.web.search import get_search_provider
    provider = get_search_provider()
    content = await provider.fetch(url, max_chars=max_chars)
    return ToolResult(
        success=content is not None,
        output={"url": url, "content": content[:max_chars] if content else ""},
        error="" if content else "Failed to fetch URL",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_agent_tools(registry: ToolRegistry) -> None:
    """Register all agent-capable tools in the unified ToolRegistry."""

    tools = [
        # ===== BROWSER =====
        Tool(
            name="browser.open",
            description="Open a URL in the browser",
            category="browser",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to open"},
                },
                "required": ["url"],
            },
            handler=browser_open,
        ),
        Tool(
            name="browser.navigate",
            description="Navigate to a URL in the browser",
            category="browser",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to navigate to"},
                },
                "required": ["url"],
            },
            handler=browser_navigate,
        ),
        Tool(
            name="browser.back",
            description="Navigate back in browser history",
            category="browser",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=browser_back,
        ),
        Tool(
            name="browser.forward",
            description="Navigate forward in browser history",
            category="browser",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=browser_forward,
        ),
        Tool(
            name="browser.click",
            description="Click an element by visible text on the current page",
            category="browser",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Visible text of the element to click"},
                },
                "required": ["text"],
            },
            handler=browser_click,
        ),
        Tool(
            name="browser.type",
            description="Type text into the currently focused element on the page",
            category="browser",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                },
                "required": ["text"],
            },
            handler=browser_type,
        ),
        Tool(
            name="browser.scroll",
            description="Scroll the browser page up or down",
            category="browser",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down"], "default": "down"},
                    "amount": {"type": "integer", "default": 3, "description": "Scroll amount"},
                },
            },
            handler=browser_scroll,
        ),
        Tool(
            name="browser.read",
            description="Read the visible text content of the current page",
            category="browser",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "default": 5000},
                },
            },
            handler=browser_read,
        ),
        Tool(
            name="browser.search",
            description="Search Google for a query via the browser",
            category="browser",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
            handler=browser_search,
        ),
        Tool(
            name="browser.get_url",
            description="Get the current browser URL",
            category="browser",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=browser_get_url,
        ),
        Tool(
            name="browser.get_title",
            description="Get the current browser page title",
            category="browser",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=browser_get_title,
        ),
        Tool(
            name="browser.close",
            description="Close the browser",
            category="browser",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=browser_close,
        ),

        # ===== DESKTOP =====
        Tool(
            name="desktop.click",
            description="Click at screen coordinates (x, y)",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "X coordinate"},
                    "y": {"type": "integer", "description": "Y coordinate"},
                },
                "required": ["x", "y"],
            },
            handler=lambda x, y: desktop_click(x, y),
        ),
        Tool(
            name="desktop.double_click",
            description="Double-click at screen coordinates",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
            handler=lambda x, y: desktop_double_click(x, y),
        ),
        Tool(
            name="desktop.right_click",
            description="Right-click at screen coordinates",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
            handler=lambda x, y: desktop_right_click(x, y),
        ),
        Tool(
            name="desktop.type",
            description="Type text using keyboard emulation",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to type"},
                },
                "required": ["text"],
            },
            handler=lambda text: desktop_type(text),
        ),
        Tool(
            name="desktop.press",
            description="Press a single key (enter, escape, tab, backspace, etc.)",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Key name to press"},
                },
                "required": ["key"],
            },
            handler=lambda key: desktop_press(key),
        ),
        Tool(
            name="desktop.hotkey",
            description="Press a keyboard shortcut (e.g., ctrl+c, alt+tab)",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keys to press together",
                    },
                },
                "required": ["keys"],
            },
            handler=lambda keys: desktop_hotkey(*keys),
        ),
        Tool(
            name="desktop.scroll",
            description="Scroll at screen coordinates",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "delta": {"type": "integer", "default": -3, "description": "Negative=down, positive=up"},
                },
                "required": ["x", "y"],
            },
            handler=lambda x, y, delta=-3: desktop_scroll(x, y, delta),
        ),
        Tool(
            name="desktop.focus_window",
            description="Find and focus a window by title substring",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Window title substring to find"},
                },
                "required": ["title"],
            },
            handler=lambda title: desktop_focus_window(title),
        ),
        Tool(
            name="desktop.open_app",
            description="Open an application by name",
            category="desktop",
            permission_level=PermissionLevel.LOW_RISK,
            parameters={
                "type": "object",
                "properties": {
                    "application": {"type": "string", "description": "Application name"},
                },
                "required": ["application"],
            },
            handler=lambda application: desktop_open_app(application),
        ),

        # ===== SCREEN =====
        Tool(
            name="screen.get_active_window",
            description="Get information about the currently active window",
            category="screen",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=screen_get_active_window,
        ),
        Tool(
            name="screen.inspect",
            description="Get a summary of the current screen state including active window and UI elements",
            category="screen",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=screen_inspect,
        ),
        Tool(
            name="screen.get_ui_tree",
            description="Get UI elements of the active window via accessibility",
            category="screen",
            permission_level=PermissionLevel.SAFE,
            parameters={"type": "object", "properties": {}},
            handler=screen_get_ui_tree,
        ),

        # ===== WEB SEARCH =====
        Tool(
            name="web_search.search",
            description="Search the web using DuckDuckGo (free, no API key required)",
            category="web_search",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            handler=web_search,
        ),
        Tool(
            name="web_search.fetch",
            description="Fetch content from a URL",
            category="web_search",
            permission_level=PermissionLevel.SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "default": 10000},
                },
                "required": ["url"],
            },
            handler=web_fetch,
        ),
    ]

    for tool in tools:
        registry.register(tool)

    logger.info("agent_tools_registered", count=len(tools))
