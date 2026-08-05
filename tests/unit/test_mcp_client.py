"""Unit tests for `agents/tools/mcp_client.py`.

`InProcessToolClient` calls plain functions directly and needs no mocking.
`MCPToolClient` talks to a real FastMCP server over HTTP — no server is
spun up here (that's the integration/e2e suite's job, and was verified
manually against a running `mcp_server.server` container), but its own
logic (result-shape coercion, exception-to-ToolCallRecord mapping) is pure
and fully testable by faking `fastmcp.Client` itself.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from agents.tools.mcp_client import InProcessToolClient, MCPToolClient
from mcp_server.tools.banking_tools import _ACCOUNTS
from schemas.dto import ToolCallRecord


class _FakeNetworkResult:
    """Stands in for fastmcp's `CallToolResult` wrapper (exposes `.data`)."""

    def __init__(self, data: dict[str, object]) -> None:
        self.data = data


class _FakeClientContext:
    """Fakes the `async with Client(url) as client: ...` shape."""

    def __init__(
        self, call_tool_return: object = None, call_tool_side_effect: Exception | None = None
    ) -> None:
        self.call_tool = AsyncMock(side_effect=call_tool_side_effect, return_value=call_tool_return)

    async def __aenter__(self) -> _FakeClientContext:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _first_account_id() -> str:
    return next(iter(_ACCOUNTS))


async def test_known_account_returns_ok_record_with_data() -> None:
    client = InProcessToolClient()
    account_id = _first_account_id()

    record = await client.call_tool("get_balance", {"account_id": account_id})

    assert isinstance(record, ToolCallRecord)
    assert record.tool_name == "get_balance"
    assert record.ok is True
    assert record.error is None
    assert record.latency_ms >= 0
    assert record.result["data"]["account_id"] == account_id


async def test_business_failure_is_surfaced_without_raising() -> None:
    client = InProcessToolClient()

    record = await client.call_tool("get_balance", {"account_id": "TR000000000000000000000000"})

    assert record.ok is False
    assert record.error == "ACCOUNT_NOT_FOUND"


async def test_unknown_tool_name_degrades_gracefully() -> None:
    client = InProcessToolClient()

    record = await client.call_tool("not_a_real_tool", {})

    assert record.ok is False
    assert record.error == "UNKNOWN_TOOL:not_a_real_tool"


async def test_bad_arguments_are_wrapped_instead_of_raising() -> None:
    client = InProcessToolClient()

    # Missing the required "account_id" kwarg -> TypeError inside get_balance.
    record = await client.call_tool("get_balance", {})

    assert record.ok is False
    assert record.error is not None


async def test_block_card_flips_status_via_the_client() -> None:
    client = InProcessToolClient()
    account_id = _first_account_id()
    card = _ACCOUNTS[account_id]["cards"][0]  # type: ignore[index]
    last4 = card["last4"]
    original_status = card["status"]

    try:
        record = await client.call_tool("block_card", {"card_last4": last4, "reason": "test"})

        assert record.ok is True
        assert record.result["data"]["status"] == "blocked"
    finally:
        card["status"] = original_status


async def test_mcp_client_coerces_a_data_wrapped_network_result() -> None:
    fake_result = _FakeNetworkResult({"ok": True, "data": {"account_id": "TR123"}})
    with patch(
        "agents.tools.mcp_client.Client", return_value=_FakeClientContext(call_tool_return=fake_result)
    ):
        client = MCPToolClient("http://mcp:8765/mcp")
        record = await client.call_tool("get_balance", {"account_id": "TR123"})

    assert record.ok is True
    assert record.result == {"ok": True, "data": {"account_id": "TR123"}}
    assert record.latency_ms >= 0


async def test_mcp_client_surfaces_a_plain_dict_network_result() -> None:
    with patch(
        "agents.tools.mcp_client.Client",
        return_value=_FakeClientContext(call_tool_return={"ok": False, "error": "ACCOUNT_NOT_FOUND"}),
    ):
        client = MCPToolClient("http://mcp:8765/mcp")
        record = await client.call_tool("get_balance", {"account_id": "TR999"})

    assert record.ok is False
    assert record.error == "ACCOUNT_NOT_FOUND"


async def test_mcp_client_never_raises_on_connection_failure() -> None:
    with patch(
        "agents.tools.mcp_client.Client",
        return_value=_FakeClientContext(call_tool_side_effect=ConnectionError("mcp unreachable")),
    ):
        client = MCPToolClient("http://mcp:8765/mcp")
        record = await client.call_tool("get_balance", {"account_id": "TR123"})

    assert record.ok is False
    assert "unreachable" in (record.error or "")
