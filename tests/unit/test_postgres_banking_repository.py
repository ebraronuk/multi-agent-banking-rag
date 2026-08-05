"""`PostgresBankingRepository` için birim testler — mocklu bir asyncpg havuzuna
karşı, gerçek bir Postgres olmadan. Amaç SQL'in doğru parametrelerle çağrıldığını
ve satır/hata eşlemesinin doğru olduğunu doğrulamak, asyncpg'nin kendisini değil.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from mcp_server.tools.banking_repository import PostgresBankingRepository


def _fake_pool(**overrides: object) -> AsyncMock:
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=overrides.get("fetchrow"))
    pool.fetchval = AsyncMock(return_value=overrides.get("fetchval"))
    pool.fetch = AsyncMock(return_value=overrides.get("fetch", []))
    return pool


def _patched_create_pool(pool: AsyncMock):
    return patch("asyncpg.create_pool", AsyncMock(return_value=pool))


async def test_get_balance_known_account() -> None:
    pool = _fake_pool(fetchrow={"balance": 12450.75, "currency": "TRY", "owner_name": "Ayşe Yılmaz (demo)"})
    with _patched_create_pool(pool):
        repo = PostgresBankingRepository("postgresql://x")
        result = await repo.get_balance("TR330006100519786457841326")

    assert result["ok"] is True
    assert result["data"]["balance"] == 12450.75
    assert result["data"]["owner_name"] == "Ayşe Yılmaz (demo)"


async def test_get_balance_unknown_account() -> None:
    pool = _fake_pool(fetchrow=None)
    with _patched_create_pool(pool):
        repo = PostgresBankingRepository("postgresql://x")
        result = await repo.get_balance("TR000000000000000000000000")

    assert result == {"ok": False, "error": "ACCOUNT_NOT_FOUND"}


async def test_get_balance_degrades_on_connection_failure() -> None:
    with patch("asyncpg.create_pool", AsyncMock(side_effect=OSError("connection refused"))):
        repo = PostgresBankingRepository("postgresql://x")
        result = await repo.get_balance("TR330006100519786457841326")

    assert result == {"ok": False, "error": "BANKING_SERVICE_UNAVAILABLE"}


async def test_list_transactions_known_account() -> None:
    rows = [
        {"occurred_at": datetime(2026, 8, 1, 9, 5, tzinfo=UTC), "description": "Maaş yatışı", "amount": 18500.0},
    ]
    pool = _fake_pool(fetchval=1, fetch=rows)
    with _patched_create_pool(pool):
        repo = PostgresBankingRepository("postgresql://x")
        result = await repo.list_transactions("TR330006100519786457841326", limit=10)

    assert result["ok"] is True
    assert result["data"]["transactions"][0]["description"] == "Maaş yatışı"
    assert result["data"]["transactions"][0]["date"] == "2026-08-01T09:05:00Z"


async def test_list_transactions_unknown_account() -> None:
    pool = _fake_pool(fetchval=None)
    with _patched_create_pool(pool):
        repo = PostgresBankingRepository("postgresql://x")
        result = await repo.list_transactions("TR000000000000000000000000")

    assert result == {"ok": False, "error": "ACCOUNT_NOT_FOUND"}


async def test_block_card_known_card() -> None:
    pool = _fake_pool(fetchval="blocked")
    with _patched_create_pool(pool):
        repo = PostgresBankingRepository("postgresql://x")
        result = await repo.block_card("4321", reason="çalındı")

    assert result["ok"] is True
    assert result["data"]["status"] == "blocked"
    assert result["data"]["reason"] == "çalındı"


async def test_block_card_unknown_card() -> None:
    pool = _fake_pool(fetchval=None)
    with _patched_create_pool(pool):
        repo = PostgresBankingRepository("postgresql://x")
        result = await repo.block_card("0000", reason="test")

    assert result == {"ok": False, "error": "CARD_NOT_FOUND"}


async def test_open_support_ticket_always_succeeds() -> None:
    repo = PostgresBankingRepository("postgresql://x")

    result = await repo.open_support_ticket(subject="Kart sorunu", description="Detaylı açıklama")

    assert result["ok"] is True
    assert result["data"]["ticket_id"].startswith("TCK-")


async def test_pool_is_created_lazily_and_only_once() -> None:
    pool = _fake_pool(fetchrow={"balance": 1.0, "currency": "TRY", "owner_name": "x"})
    create_pool = AsyncMock(return_value=pool)
    with patch("asyncpg.create_pool", create_pool):
        repo = PostgresBankingRepository("postgresql://x")
        create_pool.assert_not_called()  # __init__ havuzu henüz açmadı

        await repo.get_balance("TR1")
        await repo.get_balance("TR2")

        create_pool.assert_called_once()
