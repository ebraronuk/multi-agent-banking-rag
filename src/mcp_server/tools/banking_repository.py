"""Bankacılık veri erişimi: DATABASE_URL varsa gerçek Postgres, yoksa bellek-içi sahte veri.

`get_banking_repository(settings)` hangisinin kullanılacağına bir kere karar
verir; agents/memory.py'deki Redis/bellek-içi ayrımıyla aynı mantık.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from typing import Protocol, cast

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# InMemoryBankingRepository'nin varsayılan verisi ile db/schema.sql'deki seed
# aynı iki müşteriyi tutuyor — hangi backend'e karşı çalıştırılırsa çalıştırılsın
# testler aynı sonucu görmeli.
SEED_ACCOUNTS: dict[str, dict[str, object]] = {
    "TR330006100519786457841326": {
        "balance": 12450.75,
        "currency": "TRY",
        "owner_name": "Ayşe Yılmaz (demo)",
        "transactions": [
            {
                "date": "2026-08-03T14:22:00Z",
                "description": "Migros - market alışverişi",
                "amount": -245.50,
            },
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
            {
                "date": "2026-07-25T13:15:00Z",
                "description": "Serbest çalışma ödemesi",
                "amount": 6200.00,
            },
        ],
        "cards": [
            {"last4": "9087", "status": "active"},
            {"last4": "1122", "status": "blocked"},
        ],
    },
}


class BankingRepository(Protocol):
    async def get_balance(self, account_id: str) -> dict[str, object]: ...

    async def list_transactions(self, account_id: str, limit: int = 10) -> dict[str, object]: ...

    async def block_card(self, card_last4: str, reason: str) -> dict[str, object]: ...

    async def open_support_ticket(self, subject: str, description: str) -> dict[str, object]: ...


def _find_card(
    accounts: dict[str, dict[str, object]], card_last4: str
) -> tuple[dict[str, object], dict[str, object]] | None:
    for account in accounts.values():
        cards = cast("list[dict[str, object]]", account["cards"])
        for card in cards:
            if card["last4"] == card_last4:
                return account, card
    return None


class InMemoryBankingRepository:
    """Veritabanı yokken kullanılan yol — her örnek kendi kopyasını tutar,
    böylece bir testin `block_card` ile yaptığı mutasyon başka bir teste sızmaz."""

    def __init__(self, accounts: dict[str, dict[str, object]] | None = None) -> None:
        self._accounts = accounts if accounts is not None else copy.deepcopy(SEED_ACCOUNTS)

    async def get_balance(self, account_id: str) -> dict[str, object]:
        account = self._accounts.get(account_id)
        if account is None:
            logger.info("repo_lookup_miss", repo="in_memory", tool="get_balance", account_id=account_id)
            return {"ok": False, "error": "ACCOUNT_NOT_FOUND"}

        return {
            "ok": True,
            "data": {
                "account_id": account_id,
                "balance": account["balance"],
                "currency": account["currency"],
                "owner_name": account["owner_name"],
            },
        }

    async def list_transactions(self, account_id: str, limit: int = 10) -> dict[str, object]:
        account = self._accounts.get(account_id)
        if account is None:
            logger.info(
                "repo_lookup_miss", repo="in_memory", tool="list_transactions", account_id=account_id
            )
            return {"ok": False, "error": "ACCOUNT_NOT_FOUND"}

        transactions = account["transactions"][:limit]  # type: ignore[index]
        return {"ok": True, "data": {"account_id": account_id, "transactions": transactions}}

    async def block_card(self, card_last4: str, reason: str) -> dict[str, object]:
        found = _find_card(self._accounts, card_last4)
        if found is None:
            logger.info("repo_lookup_miss", repo="in_memory", tool="block_card", card_last4=card_last4)
            return {"ok": False, "error": "CARD_NOT_FOUND"}

        _account, card = found
        card["status"] = "blocked"
        logger.info("repo_card_blocked", repo="in_memory", card_last4=card_last4, reason=reason)
        return {"ok": True, "data": {"last4": card_last4, "status": card["status"], "reason": reason}}

    async def open_support_ticket(self, subject: str, description: str) -> dict[str, object]:
        ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
        logger.info("repo_support_ticket_opened", repo="in_memory", ticket_id=ticket_id, subject=subject)
        return {
            "ok": True,
            "data": {
                "ticket_id": ticket_id,
                "subject": subject,
                "description": description,
                "status": "open",
            },
        }


class PostgresBankingRepository:
    """Gerçek backend — `accounts`/`cards`/`transactions` tablolarına (bkz.
    db/schema.sql) parametreli SQL ile erişir.

    Havuzu `__init__`'te açamıyoruz (`asyncpg.create_pool` bir coroutine),
    o yüzden ilk kullanımda, bir kilitle korunarak kuruluyor.
    """

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url
        self._pool: object | None = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> object:
        if self._pool is None:
            async with self._pool_lock:
                if self._pool is None:
                    import asyncpg

                    self._pool = await asyncpg.create_pool(self._database_url, min_size=1, max_size=5)
        return self._pool

    async def get_balance(self, account_id: str) -> dict[str, object]:
        try:
            pool = await self._get_pool()
            row = await pool.fetchrow(  # type: ignore[attr-defined]
                "SELECT balance, currency, owner_name FROM accounts WHERE account_id = $1",
                account_id,
            )
        except Exception:
            logger.warning("repo_get_balance_failed", repo="postgres", account_id=account_id, exc_info=True)
            return {"ok": False, "error": "BANKING_SERVICE_UNAVAILABLE"}

        if row is None:
            logger.info("repo_lookup_miss", repo="postgres", tool="get_balance", account_id=account_id)
            return {"ok": False, "error": "ACCOUNT_NOT_FOUND"}

        return {
            "ok": True,
            "data": {
                "account_id": account_id,
                "balance": float(row["balance"]),
                "currency": row["currency"],
                "owner_name": row["owner_name"],
            },
        }

    async def list_transactions(self, account_id: str, limit: int = 10) -> dict[str, object]:
        try:
            pool = await self._get_pool()
            exists = await pool.fetchval("SELECT 1 FROM accounts WHERE account_id = $1", account_id)  # type: ignore[attr-defined]
            if not exists:
                logger.info(
                    "repo_lookup_miss", repo="postgres", tool="list_transactions", account_id=account_id
                )
                return {"ok": False, "error": "ACCOUNT_NOT_FOUND"}

            rows = await pool.fetch(  # type: ignore[attr-defined]
                "SELECT occurred_at, description, amount FROM transactions "
                "WHERE account_id = $1 ORDER BY occurred_at DESC LIMIT $2",
                account_id,
                limit,
            )
        except Exception:
            logger.warning(
                "repo_list_transactions_failed", repo="postgres", account_id=account_id, exc_info=True
            )
            return {"ok": False, "error": "BANKING_SERVICE_UNAVAILABLE"}

        transactions = [
            {
                "date": row["occurred_at"].isoformat().replace("+00:00", "Z"),
                "description": row["description"],
                "amount": float(row["amount"]),
            }
            for row in rows
        ]
        return {"ok": True, "data": {"account_id": account_id, "transactions": transactions}}

    async def block_card(self, card_last4: str, reason: str) -> dict[str, object]:
        try:
            pool = await self._get_pool()
            status = await pool.fetchval(  # type: ignore[attr-defined]
                "UPDATE cards SET status = 'blocked' WHERE card_last4 = $1 RETURNING status",
                card_last4,
            )
        except Exception:
            logger.warning("repo_block_card_failed", repo="postgres", card_last4=card_last4, exc_info=True)
            return {"ok": False, "error": "BANKING_SERVICE_UNAVAILABLE"}

        if status is None:
            logger.info("repo_lookup_miss", repo="postgres", tool="block_card", card_last4=card_last4)
            return {"ok": False, "error": "CARD_NOT_FOUND"}

        logger.info("repo_card_blocked", repo="postgres", card_last4=card_last4, reason=reason)
        return {"ok": True, "data": {"last4": card_last4, "status": status, "reason": reason}}

    async def open_support_ticket(self, subject: str, description: str) -> dict[str, object]:
        # Destek talebi ayrı bir sistemde (Zendesk vb.) yaşar normalde — burada
        # tablo açmak o entegrasyonu taklit etmekten öteye gitmezdi.
        ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
        logger.info("repo_support_ticket_opened", repo="postgres", ticket_id=ticket_id, subject=subject)
        return {
            "ok": True,
            "data": {
                "ticket_id": ticket_id,
                "subject": subject,
                "description": description,
                "status": "open",
            },
        }


# Veritabanı yoksa tek, süreç-genelinde bir örnek — her istekte sıfırdan bir
# InMemoryBankingRepository kurmak, bakiye/kart durumundaki değişiklikleri
# bir sonraki istekte kaybettirir.
_fallback_repository = InMemoryBankingRepository()


def get_banking_repository(settings: Settings) -> BankingRepository:
    if not settings.database_url:
        return _fallback_repository
    return PostgresBankingRepository(settings.database_url)
