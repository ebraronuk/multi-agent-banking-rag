"""`agents/memory.py` için birim testler.

`RedisMemory`, mocklu bir `redis.asyncio.Redis` istemcisine karşı test
ediliyor (gerçek Redis'e gerek yok) — amaç Redis'in kendisini değil, kendi
mantığını (key formatı, TTL aktarımı, bir depolama hatasında zarifçe düşme)
doğrulamak. `InMemoryMemory`'nin mock'lanmaya ihtiyacı yok; CI/lokal
geliştirmede kullanılan gerçek şeyin ta kendisi.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from agents.memory import (
    InMemoryMemory,
    RedisMemory,
    get_conversation_memory,
    synthesize_bare_answer_entity,
)
from app.core.config import Settings
from schemas.dto import EntityType, IntentLabel, PendingEntityRequest


async def test_in_memory_round_trips_turns_and_pending_request() -> None:
    memory = InMemoryMemory()
    pending = PendingEntityRequest(intent=IntentLabel.CARD_ACTION, entity_type=EntityType.CARD_LAST4, original_message="kartımı blokla")

    await memory.save_turn(
        "conv-1", "kartımı blokla", "hangi kartın son 4 hanesi?", pending, history_limit=6
    )
    context = await memory.load("conv-1")

    assert [t.content for t in context.turns] == ["kartımı blokla", "hangi kartın son 4 hanesi?"]
    assert context.pending_entity_request == pending


async def test_in_memory_round_trips_escalation_stage() -> None:
    memory = InMemoryMemory()

    await memory.save_turn(
        "conv-escalated",
        "insanla konuşmak istiyorum",
        "sizi aktarıyorum",
        None,
        history_limit=6,
        escalation_stage="handed_off",
    )
    context = await memory.load("conv-escalated")

    assert context.escalation_stage == "handed_off"


async def test_in_memory_escalation_stage_defaults_to_none_when_not_passed() -> None:
    memory = InMemoryMemory()

    await memory.save_turn("conv-normal", "bakiyem ne kadar", "1000 TL", None, history_limit=6)
    context = await memory.load("conv-normal")

    assert context.escalation_stage is None


async def test_in_memory_truncates_to_history_limit() -> None:
    memory = InMemoryMemory()
    for i in range(5):
        await memory.save_turn("conv-2", f"soru {i}", f"cevap {i}", None, history_limit=4)

    context = await memory.load("conv-2")

    assert len(context.turns) == 4
    assert context.turns[0].content == "soru 3"  # tutulan son 2 turn'ün en eskisi
    assert context.turns[-1].content == "cevap 4"


async def test_in_memory_unknown_conversation_returns_empty_context() -> None:
    memory = InMemoryMemory()

    context = await memory.load("never-seen")

    assert context.turns == []
    assert context.pending_entity_request is None


async def test_get_conversation_memory_falls_back_without_redis_url() -> None:
    settings = Settings(redis_url=None)

    memory = get_conversation_memory(settings)

    assert isinstance(memory, InMemoryMemory)


async def test_get_conversation_memory_returns_redis_backend_when_configured() -> None:
    settings = Settings(redis_url="redis://localhost:6379/0")

    memory = get_conversation_memory(settings)

    assert isinstance(memory, RedisMemory)


async def test_redis_memory_load_uses_the_expected_key_and_decodes() -> None:
    memory = RedisMemory("redis://localhost:6379/0", ttl_seconds=60)
    memory._client = AsyncMock()
    memory._client.get.return_value = '{"turns": [], "pending_entity_request": null}'

    context = await memory.load("conv-3")

    memory._client.get.assert_awaited_once_with("conversation:conv-3")
    assert context.turns == []


async def test_redis_memory_save_sets_with_ttl() -> None:
    memory = RedisMemory("redis://localhost:6379/0", ttl_seconds=3600)
    memory._client = AsyncMock()
    memory._client.get.return_value = None

    await memory.save_turn("conv-4", "merhaba", "merhaba, size nasıl yardımcı olabilirim?", None, history_limit=6)

    memory._client.set.assert_awaited_once()
    args, kwargs = memory._client.set.await_args
    assert args[0] == "conversation:conv-4"
    assert kwargs["ex"] == 3600


async def test_redis_memory_load_degrades_gracefully_on_connection_error() -> None:
    memory = RedisMemory("redis://localhost:6379/0", ttl_seconds=60)
    memory._client = AsyncMock()
    memory._client.get.side_effect = ConnectionError("redis unreachable")

    context = await memory.load("conv-5")

    assert context.turns == []
    assert context.pending_entity_request is None


async def test_redis_memory_save_degrades_gracefully_on_connection_error() -> None:
    memory = RedisMemory("redis://localhost:6379/0", ttl_seconds=60)
    memory._client = AsyncMock()
    memory._client.get.side_effect = ConnectionError("redis unreachable")

    # Raise etmemeli — kaybolan bir yazma bir sonraki turn'ün hafızasını
    # bozar, bunu başarısız yapmaz.
    await memory.save_turn("conv-6", "merhaba", "merhaba", None, history_limit=6)


def test_synthesize_bare_answer_entity_recognizes_a_bare_card_digit_answer() -> None:
    pending = PendingEntityRequest(intent=IntentLabel.CARD_ACTION, entity_type=EntityType.CARD_LAST4, original_message="kartımı blokla")

    result = synthesize_bare_answer_entity("1234", pending)

    assert result == (IntentLabel.CARD_ACTION, EntityType.CARD_LAST4, "1234")


def test_synthesize_bare_answer_entity_recognizes_an_iban_answer() -> None:
    pending = PendingEntityRequest(intent=IntentLabel.ACCOUNT_ACTION, entity_type=EntityType.IBAN, original_message="bakiyem ne kadar")

    result = synthesize_bare_answer_entity("TR330006100519786457841326", pending)

    assert result is not None
    intent, entity_type, value = result
    assert intent == IntentLabel.ACCOUNT_ACTION
    assert entity_type == EntityType.IBAN
    assert value == "TR330006100519786457841326"


def test_synthesize_bare_answer_entity_rejects_unrelated_text() -> None:
    pending = PendingEntityRequest(intent=IntentLabel.CARD_ACTION, entity_type=EntityType.CARD_LAST4, original_message="kartımı blokla")

    assert synthesize_bare_answer_entity("bilmiyorum", pending) is None
    # Tam 4 hane değil — bir tutar, bir yıl, başka bir şey olabilir; tahmin etme.
    assert synthesize_bare_answer_entity("12345", pending) is None
