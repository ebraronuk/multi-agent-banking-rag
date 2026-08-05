"""FastMCP server exposing DemoBank's (mocked) banking tools over the network.

This is the network-facing half of the tool-calling demo: `agents.tools.mcp_client
.MCPToolClient` talks to whatever process runs this module over HTTP, the same
way a real deployment would talk to a tool-serving process owned by a
different team. `banking_tools.py` holds the actual (fixture) business logic
so it stays independently unit-testable without FastMCP or a running server;
this module's only job is to wrap those functions as MCP tools.
"""

from __future__ import annotations

from fastmcp import FastMCP

from app.core.config import get_settings
from app.core.logging import get_logger
from mcp_server.tools.banking_tools import (
    block_card as _block_card,
)
from mcp_server.tools.banking_tools import (
    get_balance as _get_balance,
)
from mcp_server.tools.banking_tools import (
    list_transactions as _list_transactions,
)
from mcp_server.tools.banking_tools import (
    open_support_ticket as _open_support_ticket,
)

logger = get_logger(__name__)

mcp = FastMCP("demobank-tools")


@mcp.tool()
def get_balance(account_id: str) -> dict[str, object]:
    """Look up the current balance and currency for a DemoBank account by IBAN."""
    return _get_balance(account_id)


@mcp.tool()
def list_transactions(account_id: str, limit: int = 10) -> dict[str, object]:
    """List the most recent transactions for a DemoBank account by IBAN."""
    return _list_transactions(account_id, limit=limit)


@mcp.tool()
def block_card(card_last4: str, reason: str) -> dict[str, object]:
    """Block a DemoBank card identified by its last 4 digits, e.g. after a loss or theft report."""
    return _block_card(card_last4, reason=reason)


@mcp.tool()
def open_support_ticket(subject: str, description: str) -> dict[str, object]:
    """Open a human-support ticket for anything the automated tools can't resolve."""
    return _open_support_ticket(subject, description)


if __name__ == "__main__":
    settings = get_settings()
    logger.info("mcp_server_starting", host=settings.mcp_server_host, port=settings.mcp_server_port)
    # `mcp.run()` defaults to the stdio transport (no host/port); the network
    # path this project's MCPToolClient needs requires selecting the HTTP
    # transport explicitly. Verified against the pinned fastmcp version in the
    # Docker integration pass (`mcp.run(transport=...)` accepting host/port is
    # specific to "http"/"streamable-http"/"sse", not the default).
    mcp.run(transport="http", host=settings.mcp_server_host, port=settings.mcp_server_port)
