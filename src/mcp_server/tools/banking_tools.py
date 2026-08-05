"""Mocked banking backend for the DemoBank portfolio project.

Everything in this module is FIXTURE data held in an in-memory dict — there is
no real core-banking system behind it, no persistence, and no real customer
data. It exists purely to give the tool-calling agent something realistic to
call. A production integration would replace this module's internals with
calls to an actual core-banking API and keep the same function signatures.

These are plain functions (no FastMCP decorators) on purpose: `mcp_server/server.py`
is the only place that knows about FastMCP, so this module can be imported and
unit-tested (see `tests/unit/test_banking_tools.py`) without a running MCP
server or any network stack.

Every function returns `{"ok": True, "data": ...}` or `{"ok": False, "error": ...}`
instead of raising on a "not found" condition — a lookup miss is an expected,
recoverable outcome for a chat agent (ask the user for the right IBAN/card),
not a program error, so it doesn't deserve an exception and a stack trace.
"""

from __future__ import annotations

import uuid
from typing import cast

from app.core.logging import get_logger

logger = get_logger(__name__)

# Keyed by demo IBAN-style account id. Mutable in-process only (block_card
# flips `cards[i]["status"]` in place) — resets whenever the process restarts,
# which is fine for a demo with no persistence layer.
_ACCOUNTS: dict[str, dict[str, object]] = {
    "TR330006100519786457841326": {
        "balance": 12450.75,
        "currency": "TRY",
        "owner_name": "Ayşe Yılmaz (demo)",
        "transactions": [
            {"date": "2026-08-03T14:22:00Z", "description": "Migros - market alışverişi", "amount": -245.50},
            {"date": "2026-08-01T09:05:00Z", "description": "Maaş yatışı", "amount": 18500.00},
            {"date": "2026-07-29T19:47:00Z", "description": "Elektrik faturası", "amount": -412.30},
            {"date": "2026-07-27T11:10:00Z", "description": "ATM para çekme", "amount": -1000.00},
        ],
        "cards": [
            {"last4": "4321", "status": "active"},
        ],
    },
    "TR640001000000012345678901": {
        "balance": 3120.40,
        "currency": "TRY",
        "owner_name": "Mehmet Demir (demo)",
        "transactions": [
            {"date": "2026-08-02T18:30:00Z", "description": "Netflix aboneliği", "amount": -129.99},
            {"date": "2026-07-30T08:00:00Z", "description": "Kira ödemesi", "amount": -9500.00},
            {"date": "2026-07-25T13:15:00Z", "description": "Serbest çalışma ödemesi", "amount": 6200.00},
        ],
        "cards": [
            {"last4": "9087", "status": "active"},
            {"last4": "1122", "status": "blocked"},
        ],
    },
}


def _find_card(card_last4: str) -> tuple[dict[str, object], dict[str, object]] | None:
    """Locate a card by its last 4 digits across all demo accounts.

    `block_card` only takes the card's last 4 digits (mirroring how a real
    support agent would identify a card over the phone), so the lookup has to
    scan accounts rather than being a simple dict get.
    """
    for account in _ACCOUNTS.values():
        cards = cast("list[dict[str, object]]", account["cards"])
        for card in cards:
            if card["last4"] == card_last4:
                return account, card
    return None


def get_balance(account_id: str) -> dict[str, object]:
    """Return the current balance/currency/owner for a demo account."""
    account = _ACCOUNTS.get(account_id)
    if account is None:
        logger.info("tool_lookup_miss", tool="get_balance", account_id=account_id)
        return {"ok": False, "error": "ACCOUNT_NOT_FOUND"}

    logger.info("tool_lookup_hit", tool="get_balance", account_id=account_id)
    return {
        "ok": True,
        "data": {
            "account_id": account_id,
            "balance": account["balance"],
            "currency": account["currency"],
            "owner_name": account["owner_name"],
        },
    }


def list_transactions(account_id: str, limit: int = 10) -> dict[str, object]:
    """Return up to `limit` most-recent transactions for a demo account."""
    account = _ACCOUNTS.get(account_id)
    if account is None:
        logger.info("tool_lookup_miss", tool="list_transactions", account_id=account_id)
        return {"ok": False, "error": "ACCOUNT_NOT_FOUND"}

    transactions = account["transactions"][:limit]  # type: ignore[index]
    logger.info("tool_lookup_hit", tool="list_transactions", account_id=account_id, count=len(transactions))
    return {
        "ok": True,
        "data": {
            "account_id": account_id,
            "transactions": transactions,
        },
    }


def block_card(card_last4: str, reason: str) -> dict[str, object]:
    """Flip a demo card to `status="blocked"` and return a confirmation."""
    found = _find_card(card_last4)
    if found is None:
        logger.info("tool_lookup_miss", tool="block_card", card_last4=card_last4)
        return {"ok": False, "error": "CARD_NOT_FOUND"}

    _account, card = found
    card["status"] = "blocked"
    logger.info("tool_card_blocked", card_last4=card_last4, reason=reason)
    return {
        "ok": True,
        "data": {
            "last4": card_last4,
            "status": card["status"],
            "reason": reason,
        },
    }


def open_support_ticket(subject: str, description: str) -> dict[str, object]:
    """Always succeeds: mints a fake ticket id for a human-support handoff."""
    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    logger.info("tool_support_ticket_opened", ticket_id=ticket_id, subject=subject)
    return {
        "ok": True,
        "data": {
            "ticket_id": ticket_id,
            "subject": subject,
            "description": description,
            "status": "open",
        },
    }
