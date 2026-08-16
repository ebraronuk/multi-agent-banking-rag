"""`agents/tools/mcp_client.py` için birim testler.

`InProcessToolClient`, kendi `InMemoryBankingRepository` örneğini çağırır —
mock gerekmez. `MCPToolClient` gerçek bir FastMCP sunucusuna HTTP üzerinden
gider (bu, entegrasyon/e2e paketinin işi ve gerçek bir container'a karşı elle
doğrulandı) — kendi mantığı (sonuç şekli dönüştürme, exception'ı
ToolCallRecord'a eşleme) burada `fastmcp.Client`'ı sahteleyerek test ediliyor.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from agents.tools.mcp_client import InProcessToolClient, MCPToolClient, get_tool_client
from app.core.config import LLMProvider, Settings
from mcp_server.tools.banking_repository import SEED_ACCOUNTS, InMemoryBankingRepository
from schemas.dto import ToolCallRecord


class _FakeNetworkResult:
    """fastmcp'nin `CallToolResult` sarmalayıcısını (`.data` alanı) taklit eder."""

    def __init__(self, data: dict[str, object]) -> None:
        self.data = data


class _FakeClientContext:
    """`async with Client(url) as client: ...` şeklini taklit eder."""

    def __init__(
        self, call_tool_return: object = None, call_tool_side_effect: Exception | None = None
    ) -> None:
        self.call_tool = AsyncMock(side_effect=call_tool_side_effect, return_value=call_tool_return)

    async def __aenter__(self) -> _FakeClientContext:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _first_account_id() -> str:
    return next(iter(SEED_ACCOUNTS))


async def test_known_account_returns_ok_record_with_data() -> None:
    client = InProcessToolClient(InMemoryBankingRepository())
    account_id = _first_account_id()

    record = await client.call_tool("get_balance", {"account_id": account_id})

    assert isinstance(record, ToolCallRecord)
    assert record.tool_name == "get_balance"
    assert record.ok is True
    assert record.error is None
    assert record.latency_ms >= 0
    assert record.result["data"]["account_id"] == account_id


async def test_business_failure_is_surfaced_without_raising() -> None:
    client = InProcessToolClient(InMemoryBankingRepository())

    record = await client.call_tool("get_balance", {"account_id": "TR000000000000000000000000"})

    assert record.ok is False
    assert record.error == "ACCOUNT_NOT_FOUND"


async def test_unknown_tool_name_degrades_gracefully() -> None:
    client = InProcessToolClient(InMemoryBankingRepository())

    record = await client.call_tool("not_a_real_tool", {})

    assert record.ok is False
    assert record.error == "UNKNOWN_TOOL:not_a_real_tool"


async def test_bad_arguments_are_wrapped_instead_of_raising() -> None:
    client = InProcessToolClient(InMemoryBankingRepository())

    # Zorunlu "account_id" argümanı eksik -> get_balance içinde TypeError.
    record = await client.call_tool("get_balance", {})

    assert record.ok is False
    assert record.error is not None


async def test_block_card_flips_status_via_the_client() -> None:
    client = InProcessToolClient(InMemoryBankingRepository())
    account_id = _first_account_id()
    last4 = SEED_ACCOUNTS[account_id]["cards"][0]["last4"]  # type: ignore[index]

    record = await client.call_tool("block_card", {"card_last4": last4, "reason": "test"})

    assert record.ok is True
    assert record.result["data"]["status"] == "blocked"


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


def test_get_tool_client_uses_network_client_for_real_llm_by_default() -> None:
    settings = Settings(llm_provider=LLMProvider.GOOGLE, google_api_key="x")
    assert isinstance(get_tool_client(settings), MCPToolClient)


def test_get_tool_client_forces_in_process_even_with_real_llm() -> None:
    # Ayrı bir MCP süreci çalıştırmayan tek-konteynerli bir dağıtım (ör.
    # Render) için — gerçek LLM + in-process araçlar birlikte kullanılabilmeli.
    settings = Settings(llm_provider=LLMProvider.GOOGLE, google_api_key="x", force_in_process_tools=True)
    assert isinstance(get_tool_client(settings), InProcessToolClient)
