"""Handles IntentLabel.ACCOUNT_ACTION / TRANSACTION_ACTION / CARD_ACTION.

Which tool to call is decided by a plain lookup table, not the LLM: for these
three intents, precision (calling exactly the tool the intent implies, with
exactly the entity the user gave) matters more than letting a model improvise
which banking operation to run. The LLM is only used afterwards, to turn a
tool's raw result (or failure) into a natural-language reply — never to decide
whether/what to call.

Single-hop per turn is an intentional YAGNI choice for this demo: the node
always sets `tool_agent_done=True` after at most one tool call (or zero, if a
required entity is missing this turn). The supervisor's loop already supports
multi-hop tool use — a future version could set `tool_agent_done=False` here
to request a second call in the same turn (e.g. "block card, then open a
ticket") — but a single deterministic call is enough to demonstrate the
pattern without adding a planning step this demo doesn't need.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts.tool_prompt import TOOL_RESULT_SYSTEM_PROMPT
from agents.state import GraphState
from agents.tools.mcp_client import InProcessToolClient, MCPToolClient
from app.core.config import Settings
from app.core.llm import safe_ainvoke
from app.core.logging import get_logger
from schemas.dto import AgentTraceStep, Entity, EntityType, IntentLabel, ToolCallRecord

logger = get_logger(__name__)

_MISSING_IBAN_MESSAGE = (
    "Bu işlemi gerçekleştirebilmem için hesabınızın IBAN numarasına ihtiyacım var. "
    "Paylaşabilir misiniz?"
)
_MISSING_CARD_MESSAGE = (
    "Bu işlemi gerçekleştirebilmem için kartınızın son 4 hanesine ihtiyacım var. "
    "Paylaşabilir misiniz?"
)
_UNSUPPORTED_INTENT_MESSAGE = (
    "Bu talebi şu an işleyemedim. Bir müşteri temsilcisine aktarabilirim, ister misiniz?"
)

# intent -> (required entity type, tool name, message to ask for the missing entity).
# The supervisor only routes here for these three intents (see
# `agents/supervisor.py::TOOL_DRIVEN_INTENTS`), so this doubles as the single
# source of truth for "which tool answers which intent".
_INTENT_TOOL_MAP: dict[IntentLabel, tuple[EntityType, str, str]] = {
    IntentLabel.ACCOUNT_ACTION: (EntityType.IBAN, "get_balance", _MISSING_IBAN_MESSAGE),
    IntentLabel.TRANSACTION_ACTION: (EntityType.IBAN, "list_transactions", _MISSING_IBAN_MESSAGE),
    IntentLabel.CARD_ACTION: (EntityType.CARD_LAST4, "block_card", _MISSING_CARD_MESSAGE),
}


def _find_entity(entities: list[Entity], entity_type: EntityType) -> Entity | None:
    return next((e for e in entities if e.type == entity_type), None)


def _build_arguments(tool_name: str, entity_value: str, user_query: str) -> dict[str, object]:
    if tool_name in ("get_balance", "list_transactions"):
        return {"account_id": entity_value}
    if tool_name == "block_card":
        return {"card_last4": entity_value, "reason": user_query}
    raise ValueError(f"unmapped tool: {tool_name}")  # unreachable given _INTENT_TOOL_MAP


def _format_tool_outcome(record: ToolCallRecord) -> str:
    if record.ok:
        return f"Araç: {record.tool_name}\nSonuç: {record.result}"
    return f"Araç: {record.tool_name}\nHata: {record.error or 'bilinmeyen hata'}"


def build_tool_agent_node(
    tool_client: MCPToolClient | InProcessToolClient,
    llm: BaseChatModel,
    settings: Settings,
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    """Bind a tool client + LLM into an async LangGraph node.

    `settings` is accepted (rather than reaching for `get_settings()` inside
    the node) purely for dependency-injection symmetry with the other
    `build_*_node` factories in this package — nothing here currently reads
    from it, since the iteration cap itself is enforced by the supervisor.
    """

    async def tool_agent_node(state: GraphState) -> dict[str, object]:
        intent = state.get("intent")
        mapping = _INTENT_TOOL_MAP.get(intent) if intent is not None else None

        if mapping is None:
            # Defensive only: the supervisor is not expected to route here for
            # any other intent. Fail safe instead of crashing if it ever does.
            logger.warning("tool_agent_unmapped_intent", intent=intent)
            return {
                "draft_answer": _UNSUPPORTED_INTENT_MESSAGE,
                "tool_agent_done": True,
                "trace": [
                    AgentTraceStep(
                        node="tool_agent",
                        summary=f"no tool mapping for intent={intent}; short-circuited",
                    )
                ],
            }

        entity_type, tool_name, missing_message = mapping
        entities = state.get("entities", [])
        entity = _find_entity(entities, entity_type)

        if entity is None:
            # Don't call a tool with a guessed/empty value, and don't loop
            # waiting for an entity the user didn't provide this turn —
            # setting tool_agent_done=True ends the turn immediately.
            logger.info("tool_agent_missing_entity", intent=intent, entity_type=entity_type)
            return {
                "draft_answer": missing_message,
                "tool_agent_done": True,
                "trace": [
                    AgentTraceStep(
                        node="tool_agent",
                        summary=f"missing required entity {entity_type} for intent={intent}",
                        metadata={"intent": str(intent), "entity_type": str(entity_type)},
                    )
                ],
            }

        entity_value = entity.normalized or entity.value
        arguments = _build_arguments(tool_name, entity_value, state["user_query"])

        record = await tool_client.call_tool(tool_name, arguments)

        # If the summarizing LLM call itself fails (provider outage, timeout),
        # fall back to the deterministic formatter rather than an apology —
        # the tool result is already real data sitting in `record`; losing it
        # behind a generic "couldn't answer" message would throw away a
        # perfectly good answer over an unrelated LLM hiccup.
        draft_answer = await safe_ainvoke(
            llm,
            [
                SystemMessage(content=TOOL_RESULT_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Kullanıcı sorusu: {state['user_query']}\n\n{_format_tool_outcome(record)}"
                ),
            ],
            node="tool_agent",
        ) or _format_tool_outcome(record)

        logger.info(
            "tool_agent_call_completed",
            tool_name=tool_name,
            ok=record.ok,
            latency_ms=record.latency_ms,
        )

        return {
            "tool_calls": [record],
            "draft_answer": draft_answer,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "tool_agent_done": True,
            "trace": [
                AgentTraceStep(
                    node="tool_agent",
                    summary=f"called {tool_name} (ok={record.ok})",
                    metadata={"tool_name": tool_name, "ok": record.ok},
                )
            ],
        }

    return tool_agent_node
