"""Kısa süreli konuşma hafızası: her `conversation_id` için turn geçmişi +
"ne cevap bekliyorduk" slot-doldurma durumu.

Aynı duck-typed arayüzün (`async load`/`async save_turn`) arkasında iki
backend — `app/core/llm.py`/`rag/embeddings.py` ile aynı fail-open desen:
`REDIS_URL` set edilmişse `RedisMemory` (replikalar arası paylaşılır, restart'a
dayanır), yoksa `InMemoryMemory` (tek process'lik dict, lokal/test/CI için
sorun değil, restart'ta gider). İkisi de bir depolama hatasını turn'ü
çökertmeye izin vermiyor — bir Redis aksaklığı "bu turn hafızasız" olarak
zarifçe düşüyor, 500 olarak değil (bkz. ADR-008).
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.core.config import Settings
from app.core.logging import get_logger
from schemas.dto import ChatMessage, EntityType, IntentLabel, PendingEntityRequest

logger = get_logger(__name__)


class ConversationContext:
    def __init__(
        self,
        turns: list[ChatMessage] | None = None,
        pending_entity_request: PendingEntityRequest | None = None,
    ) -> None:
        self.turns = turns or []
        self.pending_entity_request = pending_entity_request


def _encode(context: ConversationContext) -> str:
    return json.dumps(
        {
            "turns": [t.model_dump(mode="json") for t in context.turns],
            "pending_entity_request": (
                context.pending_entity_request.model_dump(mode="json")
                if context.pending_entity_request
                else None
            ),
        }
    )


def _decode(raw: str) -> ConversationContext:
    data = json.loads(raw)
    pending_raw = data.get("pending_entity_request")
    return ConversationContext(
        turns=[ChatMessage(**t) for t in data.get("turns", [])],
        pending_entity_request=PendingEntityRequest(**pending_raw) if pending_raw else None,
    )


class InMemoryMemory:
    """Process-local bir dict — restart'a dayanmaz, replikalar arası paylaşılmaz,
    ama sıfır bağımlılık. Testlerin ve `LLM_PROVIDER=fake` lokal geliştirmenin
    varsayılan olarak çalıştığı yer (ADR-003'ün fail-open-offline felsefesi,
    burada LLM/embedding istemcisi yerine hafızaya uygulanmış)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def load(self, conversation_id: str) -> ConversationContext:
        raw = self._store.get(conversation_id)
        return _decode(raw) if raw else ConversationContext()

    async def save_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        pending_entity_request: PendingEntityRequest | None,
        history_limit: int,
    ) -> None:
        context = await self.load(conversation_id)
        context.turns.append(ChatMessage(role="user", content=user_message))
        context.turns.append(ChatMessage(role="assistant", content=assistant_message))
        context.turns = context.turns[-history_limit:]
        context.pending_entity_request = pending_entity_request
        self._store[conversation_id] = _encode(context)


class RedisMemory:
    """Gerçek backend: replikalar arası paylaşılır, restart'a dayanır, bir TTL
    ile sınırlı — kimsenin dönmediği bir konuşma Redis'te sonsuza dek yaşamamalı."""

    def __init__(self, redis_url: str, ttl_seconds: int) -> None:
        import redis.asyncio as redis

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _key(conversation_id: str) -> str:
        return f"conversation:{conversation_id}"

    async def load(self, conversation_id: str) -> ConversationContext:
        try:
            raw = await self._client.get(self._key(conversation_id))
        except Exception:
            logger.warning("conversation_memory_load_failed", conversation_id=conversation_id, exc_info=True)
            return ConversationContext()
        if not raw:
            return ConversationContext()
        # decode_responses=True runtime'da bunu zaten str yapıyor; tip
        # tanımı hâlâ decode-öncesi bytes | str | None'ı gösteriyor.
        return _decode(raw if isinstance(raw, str) else raw.decode())

    async def save_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        pending_entity_request: PendingEntityRequest | None,
        history_limit: int,
    ) -> None:
        try:
            context = await self.load(conversation_id)
            context.turns.append(ChatMessage(role="user", content=user_message))
            context.turns.append(ChatMessage(role="assistant", content=assistant_message))
            context.turns = context.turns[-history_limit:]
            context.pending_entity_request = pending_entity_request
            await self._client.set(
                self._key(conversation_id), _encode(context), ex=self._ttl_seconds
            )
        except Exception:
            # Kaybolan bir yazma, bir sonraki turn'ün hafızasız başlaması
            # demek — bozulmuş bir konuşma, başarısız bir işlem değil. Bu
            # exception memory_save düğümünün ötesine hiç geçmemeli.
            logger.warning("conversation_memory_save_failed", conversation_id=conversation_id, exc_info=True)


ConversationMemory = InMemoryMemory | RedisMemory

# Redis yapılandırılmamışsa süreç-genelinde tek bir InMemoryMemory örneği —
# her istekte taze bir örnek kurmak amacı tamamen boşa çıkarır (her turn boş
# bir geçmiş görürdü).
_fallback_memory = InMemoryMemory()


def get_conversation_memory(settings: Settings) -> ConversationMemory:
    if not settings.redis_url:
        return _fallback_memory
    try:
        return RedisMemory(settings.redis_url, settings.conversation_ttl_seconds)
    except Exception:
        logger.warning("redis_memory_init_failed_falling_back_to_in_memory", exc_info=True)
        return _fallback_memory


def history_to_messages(history: list[ChatMessage]) -> list[BaseMessage]:
    """LLM çağıran her worker (rag_agent/smalltalk_agent/tool_agent) tarafından
    paylaşılıyor ki çok-turlu bir takip sorusu ("ya EFT için mi?") bağlamsız
    yeni bir soru değil, bir devam gibi okunsun — tek bir dönüştürme, üç ayrı
    kopya değil."""
    return [
        HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content)
        for turn in history
    ]


def synthesize_bare_answer_entity(
    text: str, pending: PendingEntityRequest
) -> tuple[IntentLabel, EntityType, str] | None:
    """`text`, `pending`'in beklediği şeye çıplak bir cevap mı?

    "1234" gibi bir takip cevabının yakınında `nlp/ner_extractor.py`'nin
    regex'lerinin anchor'lanacağı bir "kart" kelimesi yok — tek başına
    okunduğunda bu sorun değil, ama önceki turn'ün tam olarak bunu beklediğini
    bildiğimizde yanlış olur. Kalıp dar tutuldu (tam uzunlukta rakam dizisi,
    IBAN kalıbı) — "kısa her cevap" gibi geniş bir eşleşme, yanlış bir
    bankacılık işlemini sessizce tetikler; kullanıcıya tekrar sormak daha güvenli.
    """
    stripped = text.strip()

    if pending.entity_type == EntityType.CARD_LAST4 and stripped.isdigit() and len(stripped) == 4:
        return pending.intent, EntityType.CARD_LAST4, stripped

    if pending.entity_type == EntityType.IBAN:
        from nlp.ner_extractor import extract_entities

        for entity in extract_entities(text):
            if entity.type == EntityType.IBAN:
                return pending.intent, EntityType.IBAN, entity.normalized or entity.value

    return None
