"""Short-term conversation memory: per-`conversation_id` turn history + the
"what were we waiting to hear back" slot-fill state.

Two backends behind the same duck-typed interface (`async load`/`async
save_turn`), same fail-open pattern as `app/core/llm.py`/`rag/embeddings.py`:
`RedisMemory` when `REDIS_URL` is configured (shared across replicas, survives
restarts — the real answer for a multi-instance deployment), `InMemoryMemory`
otherwise (single-process dict, fine for local dev/tests/CI, gone on restart).
Neither ever lets a storage failure crash a chat turn: a Redis blip degrades
to "no memory this turn," not a 500 (see ADR-008).
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
    """Process-local dict — no persistence across restarts, no sharing across
    replicas, but zero dependencies. This is what tests and `LLM_PROVIDER=fake`
    local dev run on by default (see ADR-003's fail-open-offline philosophy,
    applied here to memory instead of the LLM/embedding client)."""

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
    """Real backend: shared across replicas, survives restarts, bounded by a
    TTL (a conversation nobody returns to shouldn't live in Redis forever)."""

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
        # decode_responses=True on the client makes this `str` at runtime;
        # the stub's return type is still the pre-decode `bytes | str | None`.
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
            # A lost write means the next turn starts without memory — a
            # degraded conversation, not a failed one. Never let this
            # exception surface past the memory_save node.
            logger.warning("conversation_memory_save_failed", conversation_id=conversation_id, exc_info=True)


ConversationMemory = InMemoryMemory | RedisMemory

# One process-wide InMemoryMemory instance when Redis isn't configured — a
# fresh instance per request would defeat the entire point (each turn would
# see an empty history).
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
    """Shared by every LLM-calling worker (`rag_agent`/`smalltalk_agent`/
    `tool_agent`) so a multi-turn follow-up ("ya EFT için mi?") reads as a
    continuation instead of a context-free new question — one conversion,
    not three slightly-different copies."""
    return [
        HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content)
        for turn in history
    ]


def synthesize_bare_answer_entity(
    text: str, pending: PendingEntityRequest
) -> tuple[IntentLabel, EntityType, str] | None:
    """Is `text` a bare answer to what `pending` was asking for?

    A follow-up like "1234" has no "kart" keyword nearby for
    `nlp/ner_extractor.py`'s regexes to anchor on — that's fine when read in
    isolation, but wrong once we know the previous turn explicitly asked for
    exactly this. Deliberately narrow (exact-length digit strings, IBAN
    pattern) rather than "any short reply" — a wrong guess here would silently
    execute the wrong banking action, which is worse than asking again.
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
