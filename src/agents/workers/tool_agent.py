"""Handles IntentLabel.ACCOUNT_ACTION / TRANSACTION_ACTION / CARD_ACTION.

Two dispatch paths, chosen by whether a real LLM is configured (see
`build_tool_agent_node`):

- **Deterministic** (`_deterministic_tool_call`, used whenever `is_fake_model(llm)`
  is True — offline/CI/no-key): a plain lookup table decides which single tool
  to call. Precision (calling exactly the tool the intent implies, with
  exactly the entity the user gave) matters more than letting a model
  improvise which banking operation to run, and a hash-based fake model has
  no notion of "which tool applies here" anyway (see ADR-002).
- **LLM-planned** (`_reasoning_tool_call`, used with a real Anthropic/OpenAI
  model): a genuine `bind_tools` ReAct-style loop, for requests the fixed map
  can't resolve alone — "block my card AND open a support ticket about it"
  needs two tools in one turn, which no single lookup-table entry expresses.
  See ADR-009 for why this refines (not replaces) ADR-002's original stance,
  and for the argument-grounding safety check every proposed tool call has to
  pass before it's allowed to execute.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as lc_tool

from agents.memory import history_to_messages
from agents.prompts.tool_prompt import TOOL_REASONING_SYSTEM_PROMPT, TOOL_RESULT_SYSTEM_PROMPT
from agents.state import GraphState
from agents.tools.mcp_client import InProcessToolClient, MCPToolClient
from app.core.config import Settings
from app.core.llm import is_fake_model, safe_ainvoke, safe_ainvoke_message
from app.core.logging import get_logger
from schemas.dto import (
    AgentTraceStep,
    Entity,
    EntityType,
    IntentLabel,
    PendingEntityRequest,
    ToolCallRecord,
)

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
_UNGROUNDED_ARGUMENT_MESSAGE = (
    "Bu işlem için verdiğiniz bilgiyi konuşmamızda doğrulayamadım. Lütfen IBAN'ınızı veya "
    "kartınızın son 4 hanesini tekrar, açıkça paylaşır mısınız?"
)

# intent -> (required entity type, tool name, message to ask for the missing entity).
# The supervisor only routes here for these three intents (see
# `agents/supervisor.py::TOOL_DRIVEN_INTENTS`), so this doubles as the single
# source of truth for "which tool answers which intent" in the deterministic path.
_INTENT_TOOL_MAP: dict[IntentLabel, tuple[EntityType, str, str]] = {
    IntentLabel.ACCOUNT_ACTION: (EntityType.IBAN, "get_balance", _MISSING_IBAN_MESSAGE),
    IntentLabel.TRANSACTION_ACTION: (EntityType.IBAN, "list_transactions", _MISSING_IBAN_MESSAGE),
    IntentLabel.CARD_ACTION: (EntityType.CARD_LAST4, "block_card", _MISSING_CARD_MESSAGE),
}

_MAX_TOOL_CALLS_PER_HOP = 3


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


async def _deterministic_tool_call(
    state: GraphState,
    tool_client: MCPToolClient | InProcessToolClient,
    llm: BaseChatModel,
) -> dict[str, object]:
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

    # `mapping` only comes from a successful `_INTENT_TOOL_MAP.get(intent)`
    # lookup above, which is only attempted when `intent is not None` — so
    # this always holds; the assert narrows the type for mypy.
    assert intent is not None
    entity_type, tool_name, missing_message = mapping
    entities = state.get("entities", [])
    entity = _find_entity(entities, entity_type)

    if entity is None:
        # Don't call a tool with a guessed/empty value, and don't loop
        # waiting for an entity the user didn't provide this turn —
        # setting tool_agent_done=True ends the turn immediately.
        # `pending_entity_request` is what lets *next* turn's bare "1234"
        # complete this request instead of being reclassified from scratch
        # (see memory_agent.py / ADR-008).
        logger.info("tool_agent_missing_entity", intent=intent, entity_type=entity_type)
        return {
            "draft_answer": missing_message,
            "tool_agent_done": True,
            "pending_entity_request": PendingEntityRequest(
                intent=intent, entity_type=entity_type, original_message=state["user_query"]
            ),
            "trace": [
                AgentTraceStep(
                    node="tool_agent",
                    summary=f"missing required entity {entity_type} for intent={intent}",
                    metadata={"intent": str(intent), "entity_type": str(entity_type)},
                )
            ],
        }

    entity_value = entity.normalized or entity.value
    # If this turn is completing a slot-fill ("4321" answering "which card?"),
    # the *original* request ("kartımı blokla, çalındı") is what's actually
    # worth recording as `reason` — the bare digits answer alone would be.
    carried = state.get("carried_pending_request")
    reason_source = carried.original_message if carried else state["user_query"]
    arguments = _build_arguments(tool_name, entity_value, reason_source)

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


def _grounded_entity_values(entities: list[Entity]) -> dict[EntityType, set[str]]:
    grounded: dict[EntityType, set[str]] = {}
    for entity in entities:
        grounded.setdefault(entity.type, set()).add(entity.normalized or entity.value)
    return grounded


def _validate_tool_args(
    tool_name: str, args: dict[str, object], grounded: dict[EntityType, set[str]]
) -> str | None:
    """Refuse to execute a tool call whose financial identifier the model
    invented instead of using one actually present in the conversation.

    A `bind_tools` model is free to fill in *any* string for `account_id`/
    `card_last4` — nothing stops it from hallucinating a plausible-looking
    one. This is the one check that stands between "the model reasoned about
    which tool to call" (fine, that's the point) and "the model reasoned
    about *whose account* to call it on" (never fine, see ADR-009).

    Returns an error string to feed back to the model if the argument isn't
    grounded, or None if the call may proceed.
    """
    if tool_name in ("get_balance", "list_transactions"):
        account_id = str(args.get("account_id", ""))
        if account_id not in grounded.get(EntityType.IBAN, set()):
            return "ARGUMENT_NOT_GROUNDED: account_id was not found in the conversation as a real IBAN entity"
    if tool_name == "block_card":
        card_last4 = str(args.get("card_last4", ""))
        if card_last4 not in grounded.get(EntityType.CARD_LAST4, set()):
            return "ARGUMENT_NOT_GROUNDED: card_last4 was not found in the conversation as a real card entity"
    return None


def _build_tool_specs(
    tool_client: MCPToolClient | InProcessToolClient,
    grounded: dict[EntityType, set[str]],
    collected: list[ToolCallRecord],
) -> list[BaseTool]:
    """Build the LangChain tool objects `bind_tools` needs.

    Each closes over `tool_client` so its body is the *real* execution path
    (through the same MCP client every other path uses, not a stub) —
    argument grounding is checked first, and every call (successful, refused,
    or failed) is appended to `collected` so the API's `tool_calls` field
    reflects what the reasoning loop actually did, not just what the
    deterministic path would have.
    """

    async def _call(tool_name: str, args: dict[str, object]) -> str:
        violation = _validate_tool_args(tool_name, args, grounded)
        if violation:
            collected.append(
                ToolCallRecord(tool_name=tool_name, arguments=args, ok=False, error=violation, latency_ms=0.0)
            )
            logger.warning("tool_agent_ungrounded_argument", tool_name=tool_name, args=args)
            return violation
        record = await tool_client.call_tool(tool_name, args)
        collected.append(record)
        return _format_tool_outcome(record)

    @lc_tool
    async def get_balance(account_id: str) -> str:
        """Look up the current balance and currency for a DemoBank account by IBAN."""
        return await _call("get_balance", {"account_id": account_id})

    @lc_tool
    async def list_transactions(account_id: str, limit: int = 10) -> str:
        """List the most recent transactions for a DemoBank account by IBAN."""
        return await _call("list_transactions", {"account_id": account_id, "limit": limit})

    @lc_tool
    async def block_card(card_last4: str, reason: str) -> str:
        """Block a DemoBank card by its last 4 digits, e.g. after a loss or theft report."""
        return await _call("block_card", {"card_last4": card_last4, "reason": reason})

    @lc_tool
    async def open_support_ticket(subject: str, description: str) -> str:
        """Open a human support ticket for anything the other tools can't resolve."""
        return await _call("open_support_ticket", {"subject": subject, "description": description})

    return [get_balance, list_transactions, block_card, open_support_ticket]


async def _reasoning_tool_call(
    state: GraphState,
    tool_client: MCPToolClient | InProcessToolClient,
    llm: BaseChatModel,
    settings: Settings,
) -> dict[str, object]:
    entities = state.get("entities", [])
    grounded = _grounded_entity_values(entities)
    collected: list[ToolCallRecord] = []
    tools = _build_tool_specs(tool_client, grounded, collected)
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    entity_hints = ", ".join(f"{e.type}={e.normalized or e.value}" for e in entities) or "yok"
    messages: list[BaseMessage] = [
        SystemMessage(content=TOOL_REASONING_SYSTEM_PROMPT),
        *history_to_messages(state.get("history", [])),
        HumanMessage(
            content=f"Konuşmada tespit edilen bilgiler: {entity_hints}\n\nKullanıcı: {state['user_query']}"
        ),
    ]

    trace: list[AgentTraceStep] = []
    final_text: str | None = None
    max_hops = settings.max_agent_iterations

    for hop in range(max_hops):
        ai_message = await safe_ainvoke_message(llm_with_tools, messages, node="tool_agent")
        if ai_message is None:
            break  # provider failure — fall through to whatever we've collected so far

        messages.append(ai_message)
        tool_calls = ai_message.tool_calls[:_MAX_TOOL_CALLS_PER_HOP]

        if not tool_calls:
            final_text = str(ai_message.content)
            trace.append(
                AgentTraceStep(node="tool_agent", summary=f"reasoning loop concluded after {hop} hop(s)")
            )
            break

        called_names = []
        for call in tool_calls:
            tool_obj = tools_by_name.get(call["name"])
            if tool_obj is None:
                result_text = f"UNKNOWN_TOOL:{call['name']}"
                logger.warning("tool_agent_hallucinated_tool_name", tool_name=call["name"])
            else:
                result_text = await tool_obj.ainvoke(call["args"])
            messages.append(ToolMessage(content=result_text, tool_call_id=call["id"]))
            called_names.append(call["name"])

        trace.append(
            AgentTraceStep(
                node="tool_agent", summary=f"hop {hop + 1}: called {called_names}", metadata={"hop": hop + 1}
            )
        )
    else:
        trace.append(AgentTraceStep(node="tool_agent", summary="reasoning loop hit max hops"))

    if final_text is None:
        final_text = (
            _format_tool_outcome(collected[-1]) if collected else _UNSUPPORTED_INTENT_MESSAGE
        )

    return {
        "tool_calls": collected,
        "draft_answer": final_text,
        "iteration_count": state.get("iteration_count", 0) + len(collected),
        "tool_agent_done": True,
        "trace": trace,
    }


def build_tool_agent_node(
    tool_client: MCPToolClient | InProcessToolClient,
    llm: BaseChatModel,
    settings: Settings,
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    """Bind a tool client + LLM into an async LangGraph node.

    Dispatches to the deterministic single-tool path or the LLM-planned
    multi-tool reasoning loop based on whether `llm` is the offline
    `FakeChatModel` — decided once here (not per-request) since it can't
    change mid-process.
    """
    use_reasoning = not is_fake_model(llm)

    async def tool_agent_node(state: GraphState) -> dict[str, object]:
        if use_reasoning:
            return await _reasoning_tool_call(state, tool_client, llm, settings)
        return await _deterministic_tool_call(state, tool_client, llm)

    return tool_agent_node
