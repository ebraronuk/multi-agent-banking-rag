"""Unit tests for the mocked banking backend (`mcp_server/tools/banking_tools.py`).

Pure in-memory fixture data, no network/DB/FastMCP involved — these exercise
the found/not-found contract each tool promises, plus the one function with
a real side effect (`block_card` mutating shared state).
"""

from __future__ import annotations

from mcp_server.tools.banking_tools import (
    _ACCOUNTS,
    block_card,
    get_balance,
    list_transactions,
    open_support_ticket,
)

_UNKNOWN_ACCOUNT_ID = "TR000000000000000000000000"
_UNKNOWN_CARD_LAST4 = "0000"


def _first_account_id() -> str:
    return next(iter(_ACCOUNTS))


def _first_card_last4(account_id: str) -> str:
    return _ACCOUNTS[account_id]["cards"][0]["last4"]  # type: ignore[index]


class TestGetBalance:
    def test_known_account_returns_balance_data(self) -> None:
        account_id = _first_account_id()
        result = get_balance(account_id)

        assert result["ok"] is True
        data = result["data"]
        assert data["account_id"] == account_id
        assert data["currency"] == _ACCOUNTS[account_id]["currency"]
        assert data["balance"] == _ACCOUNTS[account_id]["balance"]

    def test_unknown_account_returns_not_found(self) -> None:
        result = get_balance(_UNKNOWN_ACCOUNT_ID)

        assert result == {"ok": False, "error": "ACCOUNT_NOT_FOUND"}


class TestListTransactions:
    def test_known_account_returns_transactions_respecting_limit(self) -> None:
        account_id = _first_account_id()

        result = list_transactions(account_id, limit=2)

        assert result["ok"] is True
        assert len(result["data"]["transactions"]) == 2

    def test_unknown_account_returns_not_found(self) -> None:
        result = list_transactions(_UNKNOWN_ACCOUNT_ID)

        assert result == {"ok": False, "error": "ACCOUNT_NOT_FOUND"}


class TestBlockCard:
    def test_known_card_flips_status_to_blocked(self) -> None:
        account_id = _first_account_id()
        last4 = _first_card_last4(account_id)
        card = next(c for c in _ACCOUNTS[account_id]["cards"] if c["last4"] == last4)  # type: ignore[union-attr]
        original_status = card["status"]

        try:
            result = block_card(last4, reason="kart kayboldu")

            assert result["ok"] is True
            assert result["data"]["status"] == "blocked"
            assert card["status"] == "blocked"  # actually mutated in-memory, not just echoed back
        finally:
            card["status"] = original_status  # keep the fixture idempotent for other tests

    def test_unknown_card_returns_not_found(self) -> None:
        result = block_card(_UNKNOWN_CARD_LAST4, reason="test")

        assert result == {"ok": False, "error": "CARD_NOT_FOUND"}


class TestOpenSupportTicket:
    def test_always_succeeds_with_a_ticket_id(self) -> None:
        result = open_support_ticket(subject="Kart sorunu", description="Detaylı açıklama")

        assert result["ok"] is True
        assert result["data"]["ticket_id"].startswith("TCK-")
        assert result["data"]["status"] == "open"
