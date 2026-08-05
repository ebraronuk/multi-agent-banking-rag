"""DemoBank'ın (sahte) bankacılık araçlarını ağ üzerinden sunan FastMCP sunucusu.

`agents.tools.mcp_client.MCPToolClient` bu modülü çalıştıran süreçle HTTP
üzerinden konuşur — gerçek bir dağıtımda araçların başka bir ekibin sahip
olduğu ayrı bir servis olarak yaşadığı senaryoyu taklit eder. Asıl veri
erişimi banking_repository.py'de; bu modülün tek işi o metodları MCP aracı
olarak sarmalamak.

Repository süreç başında bir kere kuruluyor (Postgres bağlantı havuzu ilk
sorguda tembelce açılıyor, bkz. banking_repository.py) — her çağrı için değil.
"""

from __future__ import annotations

from fastmcp import FastMCP

from app.core.config import get_settings
from app.core.logging import get_logger
from mcp_server.tools.banking_repository import get_banking_repository

logger = get_logger(__name__)

mcp = FastMCP("demobank-tools")
_repository = get_banking_repository(get_settings())


@mcp.tool()
async def get_balance(account_id: str) -> dict[str, object]:
    """IBAN'a göre bir DemoBank hesabının güncel bakiyesini ve para birimini getirir."""
    return await _repository.get_balance(account_id)


@mcp.tool()
async def list_transactions(account_id: str, limit: int = 10) -> dict[str, object]:
    """Bir DemoBank hesabının en son işlemlerini IBAN'a göre listeler."""
    return await _repository.list_transactions(account_id, limit=limit)


@mcp.tool()
async def block_card(card_last4: str, reason: str) -> dict[str, object]:
    """Son 4 hanesiyle belirtilen bir DemoBank kartını bloklar (kayıp/çalıntı bildirimi vb.)."""
    return await _repository.block_card(card_last4, reason=reason)


@mcp.tool()
async def open_support_ticket(subject: str, description: str) -> dict[str, object]:
    """Otomatik araçların çözemediği durumlar için bir destek talebi açar."""
    return await _repository.open_support_ticket(subject, description)


if __name__ == "__main__":
    settings = get_settings()
    logger.info("mcp_server_starting", host=settings.mcp_server_host, port=settings.mcp_server_port)
    # mcp.run() varsayılan olarak stdio transport kullanır; MCPToolClient'ın
    # ihtiyaç duyduğu ağ yolu için HTTP transport'u açıkça seçmek gerekiyor.
    mcp.run(transport="http", host=settings.mcp_server_host, port=settings.mcp_server_port)
