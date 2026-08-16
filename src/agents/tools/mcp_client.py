"""Bankacılık araçlarını çağıran istemciler.

`MCPToolClient` gerçek bir FastMCP sunucusuna ağ üzerinden gider,
`InProcessToolClient` aynı BankingRepository metodlarını süreç içinde çağırır
(ağ yok, sunucu yok — offline/CI ve testler için). İkisi de aynı
`async call_tool(...) -> ToolCallRecord` imzasını taşıdığı için tool_agent.py
hangisini tuttuğuna hiç bakmaz.

Bir aracın çağrısı ne şekilde başarısız olursa olsun (bağlantı hatası, bilinmeyen
araç, yanlış argüman), `call_tool` exception fırlatmaz — sohbeti 500'e düşürmek
yerine bir ToolCallRecord(ok=False) döner.
"""

from __future__ import annotations

import time

from fastmcp import Client

from app.core.config import LLMProvider, Settings
from app.core.logging import get_logger
from mcp_server.tools.banking_repository import BankingRepository, get_banking_repository
from schemas.dto import ToolCallRecord

logger = get_logger(__name__)


def _extract_business_outcome(result: object) -> tuple[bool, str | None]:
    """Aracın kendi `{"ok": ..., "error": ...}` zarfını okur.

    Exception fırlatmadan dönen ama ACCOUNT_NOT_FOUND gibi bir hata bildiren
    bir çağrı da iş kuralı düzeyinde bir başarısızlıktır, tool_agent bunu
    bilmeli.
    """
    if isinstance(result, dict) and "ok" in result:
        ok = bool(result["ok"])
        return ok, (None if ok else str(result.get("error")))
    return True, None


def _coerce_network_result(result: object) -> dict[str, object] | str:
    """fastmcp'nin `call_tool` dönüşünü (sürüme göre düz dict ya da
    `CallToolResult` sarmalayıcısı) JSON-dostu bir hale indirger."""
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
    """Ağ üzerinden, çalışan bir FastMCP sunucusuna karşı çağrı yapar —
    gerçek bir dağıtımda araçların ayrı bir servis olarak yaşadığı yol budur."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url

    async def call_tool(self, name: str, arguments: dict[str, object]) -> ToolCallRecord:
        start = time.perf_counter()
        try:
            async with Client(self.server_url) as client:
                raw_result = await client.call_tool(name, arguments)
        except Exception as exc:  # noqa: BLE001 - sohbeti düşür, isteği değil
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "mcp_tool_call_failed", tool_name=name, arguments=arguments, error=str(exc)
            )
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


# Araç adı -> BankingRepository metod adı. mcp_server/server.py'deki
# @mcp.tool() adlarıyla birebir aynı stringler.
_TOOL_NAMES = ("get_balance", "list_transactions", "block_card", "open_support_ticket")


class InProcessToolClient:
    """Offline/fake yol: BankingRepository metodlarını doğrudan çağırır, ağ yok.

    `resolved_llm_provider()` FAKE olduğunda (CI, anahtarsız lokal geliştirme,
    testler) kullanılır — çalışan bir MCP sunucusu veya gerçek bir LLM anahtarı
    olmadan tüm grafiği çalıştırmayı sağlar.
    """

    def __init__(self, repository: BankingRepository) -> None:
        self._repository = repository

    async def call_tool(self, name: str, arguments: dict[str, object]) -> ToolCallRecord:
        start = time.perf_counter()
        method = getattr(self._repository, name, None) if name in _TOOL_NAMES else None
        if method is None:
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
            result = await method(**arguments)
        except Exception as exc:  # noqa: BLE001 - MCPToolClient ile aynı sözleşme
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
    """Fake LLM'de her zaman in-process istemci; gerçek LLM'de de
    `force_in_process_tools` set edilmişse (ör. ayrı bir MCP süreci
    çalıştırmayan tek-konteynerli bir dağıtım) yine in-process, aksi halde
    ağ üzerinden bir MCP sunucusuna gider.
    """
    if settings.resolved_llm_provider() == LLMProvider.FAKE or settings.force_in_process_tools:
        return InProcessToolClient(get_banking_repository(settings))
    return MCPToolClient(settings.mcp_server_url)
