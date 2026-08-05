"""Client-side adapters for calling banking tools.

Two duck-type-compatible clients (`async def call_tool(...) -> ToolCallRecord`)
so `agents/workers/tool_agent.py` never has to branch on which one it holds:

- `MCPToolClient` calls a real, network-reachable FastMCP server (`mcp_server/server.py`).
- `InProcessToolClient` calls the same plain functions in `banking_tools.py`
  directly, with no network and no FastMCP involved — used for offline/CI runs
  (`LLM_PROVIDER=fake`) and unit tests where no server is running.

Neither client ever lets a tool failure propagate as an exception out of
`call_tool`: a broken/unreachable tool must degrade the conversation (the agent
apologizes and asks for what it needs) rather than 500 the whole chat request.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from fastmcp import Client

from app.core.config import LLMProvider, Settings
from app.core.logging import get_logger
from mcp_server.tools.banking_tools import (
    block_card,
    get_balance,
    list_transactions,
    open_support_ticket,
)
from schemas.dto import ToolCallRecord

logger = get_logger(__name__)


def _extract_business_outcome(result: object) -> tuple[bool, str | None]:
    """Read the tool's own `{"ok": ..., "error": ...}` envelope, if present.

    `ToolCallRecord.ok` is what `tool_agent.py` and `tool_prompt.py` branch on
    to decide "did we get real data back" — a tool that ran without raising but
    reported e.g. ACCOUNT_NOT_FOUND is still a business-level failure the agent
    needs to know about, not a success with an unusual payload.
    """
    if isinstance(result, dict) and "ok" in result:
        ok = bool(result["ok"])
        return ok, (None if ok else str(result.get("error")))
    return True, None


def _coerce_network_result(result: object) -> dict[str, object] | str:
    """Normalize whatever `fastmcp.Client.call_tool` hands back into JSON-friendly data.

    fastmcp's return shape for `call_tool` has moved between a plain dict and a
    `CallToolResult` wrapper (exposing `.data` and/or `.content`) across
    versions; this defensively unwraps the common shapes instead of assuming
    one. Flagged as an open question for the Docker-based integration pass,
    where the installed fastmcp version can actually be exercised.
    """
    if isinstance(result, dict):
        return result
    data = getattr(result, "data", None)
    if isinstance(data, dict):
        return data
    content = getattr(result, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if text is not None:
            return str(text)
    return str(result)


class MCPToolClient:
    """Calls banking tools over the network against a running FastMCP server.

    This is the path a real deployment would use once tools live behind an
    actual service boundary (e.g. a separate core-banking gateway team owns
    `mcp_server`).
    """

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def call_tool(self, name: str, arguments: dict[str, object]) -> ToolCallRecord:
        start = time.perf_counter()
        try:
            async with Client(self.server_url) as client:
                raw_result = await client.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 - degrade the conversation, never 500 the request
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("mcp_tool_call_failed", tool_name=name, arguments=arguments, error=str(exc))
            return ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                result=None,
                ok=False,
                error=str(exc),
                latency_ms=latency_ms,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        result = _coerce_network_result(raw_result)
        ok, business_error = _extract_business_outcome(result)
        logger.info("mcp_tool_call_completed", tool_name=name, ok=ok, latency_ms=latency_ms)
        return ToolCallRecord(
            tool_name=name,
            arguments=arguments,
            result=result,
            ok=ok,
            error=business_error,
            latency_ms=latency_ms,
        )


# Tool name -> plain function in banking_tools.py. Kept as a flat dict rather
# than a generic plugin/registry framework — four fixed demo tools don't need
# dynamic discovery, and the identical string keys here and in
# `mcp_server/server.py`'s `@mcp.tool()` names are what let `tool_agent.py`
# call either client with the same tool name.
TOOL_REGISTRY: dict[str, Callable[..., dict[str, object]]] = {
    "get_balance": get_balance,
    "list_transactions": list_transactions,
    "block_card": block_card,
    "open_support_ticket": open_support_ticket,
}


class InProcessToolClient:
    """Offline/fake variant: calls the plain tool functions directly, no network.

    Used whenever `resolved_llm_provider()` is FAKE (CI, local dev without
    credentials, unit tests) so the whole graph is exercisable without a
    running MCP server or a real LLM key.
    """

    async def call_tool(self, name: str, arguments: dict[str, object]) -> ToolCallRecord:
        start = time.perf_counter()
        func = TOOL_REGISTRY.get(name)
        if func is None:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("in_process_tool_unknown", tool_name=name)
            return ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                result=None,
                ok=False,
                error=f"UNKNOWN_TOOL:{name}",
                latency_ms=latency_ms,
            )

        try:
            result = func(**arguments)
        except Exception as exc:  # noqa: BLE001 - same degrade-gracefully contract as MCPToolClient
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("in_process_tool_call_failed", tool_name=name, error=str(exc))
            return ToolCallRecord(
                tool_name=name,
                arguments=arguments,
                result=None,
                ok=False,
                error=str(exc),
                latency_ms=latency_ms,
            )

        latency_ms = (time.perf_counter() - start) * 1000
        ok, business_error = _extract_business_outcome(result)
        logger.info("in_process_tool_call_completed", tool_name=name, ok=ok, latency_ms=latency_ms)
        return ToolCallRecord(
            tool_name=name,
            arguments=arguments,
            result=result,
            ok=ok,
            error=business_error,
            latency_ms=latency_ms,
        )


def get_tool_client(settings: Settings) -> MCPToolClient | InProcessToolClient:
    """Fake-LLM runs get the in-process client; real-LLM runs get the network client.

    This split is a deliberate demo trade-off, not just a fake/real toggle: a
    real production deployment might still prefer the in-process path even
    with a real LLM configured, purely for latency (it skips an HTTP
    round-trip to `mcp_server`). Here the fake/real LLM setting doubles as the
    switch so this portfolio project demonstrably exercises both the
    network-MCP path and the fast in-process fallback path, rather than
    picking one and hiding the trade-off.
    """
    if settings.resolved_llm_provider() == LLMProvider.FAKE:
        return InProcessToolClient()
    return MCPToolClient(settings.mcp_server_url)
