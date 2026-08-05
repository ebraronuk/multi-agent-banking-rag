"""Unit tests for `agents/tools/mcp_client.py`, `InProcessToolClient` only.

`MCPToolClient`'s real network path needs an actual FastMCP server running,
which isn't available in a unit test — that path is exercised in the
integration/e2e suite instead. `InProcessToolClient` calls the same plain
functions in `banking_tools.py` directly, so it's fully testable offline.
"""

from __future__ import annotations

from agents.tools.mcp_client import InProcessToolClient
from mcp_server.tools.banking_tools import _ACCOUNTS
from schemas.dto import ToolCallRecord


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
