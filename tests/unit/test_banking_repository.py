"""`InMemoryBankingRepository` için birim testler — ağ, FastMCP veya gerçek
Postgres gerektirmez. `PostgresBankingRepository` mocklu asyncpg ile
`test_postgres_banking_repository.py`'de ayrıca test ediliyor.
"""

from __future__ import annotations

from mcp_server.tools.banking_repository import SEED_ACCOUNTS, InMemoryBankingRepository

_UNKNOWN_ACCOUNT_ID = "TR000000000000000000000000"
_UNKNOWN_CARD_LAST4 = "0000"


def _first_account_id() -> str:
    return next(iter(SEED_ACCOUNTS))


class TestGetBalance:
    async def test_known_account_returns_balance_data(self) -> None:
        repo = InMemoryBankingRepository()
        account_id = _first_account_id()

        result = await repo.get_balance(account_id)

        assert result["ok"] is True
        data = result["data"]
        assert data["account_id"] == account_id
        assert data["currency"] == SEED_ACCOUNTS[account_id]["currency"]
        assert data["balance"] == SEED_ACCOUNTS[account_id]["balance"]

    async def test_unknown_account_returns_not_found(self) -> None:
        repo = InMemoryBankingRepository()

        result = await repo.get_balance(_UNKNOWN_ACCOUNT_ID)

        assert result == {"ok": False, "error": "ACCOUNT_NOT_FOUND"}


class TestListTransactions:
    async def test_known_account_returns_transactions_respecting_limit(self) -> None:
        repo = InMemoryBankingRepository()
        account_id = _first_account_id()

        result = await repo.list_transactions(account_id, limit=2)

        assert result["ok"] is True
        assert len(result["data"]["transactions"]) == 2

    async def test_unknown_account_returns_not_found(self) -> None:
        repo = InMemoryBankingRepository()

        result = await repo.list_transactions(_UNKNOWN_ACCOUNT_ID)

        assert result == {"ok": False, "error": "ACCOUNT_NOT_FOUND"}


class TestBlockCard:
    async def test_known_card_flips_status_to_blocked(self) -> None:
        repo = InMemoryBankingRepository()
        account_id = _first_account_id()
        last4 = SEED_ACCOUNTS[account_id]["cards"][0]["last4"]  # type: ignore[index]

        result = await repo.block_card(last4, reason="kart kayboldu")

        assert result["ok"] is True
        assert result["data"]["status"] == "blocked"

    async def test_unknown_card_returns_not_found(self) -> None:
        repo = InMemoryBankingRepository()

        result = await repo.block_card(_UNKNOWN_CARD_LAST4, reason="test")

        assert result == {"ok": False, "error": "CARD_NOT_FOUND"}

    async def test_instances_do_not_share_mutable_state(self) -> None:
        # SEED_ACCOUNTS'un derin kopyası her örneğe özel olmalı — biri
        # block_card çağırdığında diğerini etkilememeli.
        account_id = _first_account_id()
        last4 = SEED_ACCOUNTS[account_id]["cards"][0]["last4"]  # type: ignore[index]
        repo_a = InMemoryBankingRepository()
        repo_b = InMemoryBankingRepository()

        await repo_a.block_card(last4, reason="test")

        result_b = await repo_b.get_balance(account_id)
        assert result_b["ok"] is True  # repo_b hâlâ bozulmamış durumda


class TestOpenSupportTicket:
    async def test_always_succeeds_with_a_ticket_id(self) -> None:
        repo = InMemoryBankingRepository()

        result = await repo.open_support_ticket(subject="Kart sorunu", description="Detaylı açıklama")

        assert result["ok"] is True
        assert result["data"]["ticket_id"].startswith("TCK-")
        assert result["data"]["status"] == "open"
