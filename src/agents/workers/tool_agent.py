"""IntentLabel.ACCOUNT_ACTION / TRANSACTION_ACTION / CARD_ACTION'ı işler.

İki yol var (`build_tool_agent_node` seçiyor): `_deterministic_tool_call`
(fake model — sabit arama tablosu, tek araç, bkz. ADR-002) ve
`_reasoning_tool_call` (gerçek LLM — `bind_tools` ile çoklu araç çağrısı,
argüman-doğrulama güvenlik kontrolü dahil, bkz. ADR-009).
"""

from __future__ import annotations

import json
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
from app.core.llm import content_to_text, is_fake_model, safe_ainvoke, safe_ainvoke_message
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

# intent -> (gereken entity tipi, araç adı, eksik entity için sorulacak mesaj).
# Deterministik yol için "hangi araç hangi niyeti yanıtlıyor" tek kaynağı.
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
    raise ValueError(f"unmapped tool: {tool_name}")  # _INTENT_TOOL_MAP göz önüne alınca ulaşılamaz


def _format_tool_outcome(record: ToolCallRecord) -> str:
    """Özetleme çağrısı başarısız olursa kullanıcıya doğrudan gösterilebilir —
    dict sonuçlar bu yüzden Python repr değil JSON ile formatlanıyor."""
    if record.ok:
        result = (
            json.dumps(record.result, ensure_ascii=False)
            if isinstance(record.result, dict)
            else record.result
        )
        return f"Araç: {record.tool_name}\nSonuç: {result}"
    return f"Araç: {record.tool_name}\nHata: {record.error or 'bilinmeyen hata'}"


async def _deterministic_tool_call(
    state: GraphState,
    tool_client: MCPToolClient | InProcessToolClient,
    llm: BaseChatModel,
) -> dict[str, object]:
    intent = state.get("intent")
    mapping = _INTENT_TOOL_MAP.get(intent) if intent is not None else None

    if mapping is None:
        # Savunma amaçlı: supervisor bu düğüme başka bir intent için yönlendirmemeli.
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

    assert intent is not None  # mapping bulunduysa intent zaten None değildi; mypy için
    entity_type, tool_name, missing_message = mapping
    entities = state.get("entities", [])
    entity = _find_entity(entities, entity_type)

    if entity is None:
        # `pending_entity_request`, bir sonraki turn'ün çıplak cevabının (ör. "1234")
        # sıfırdan sınıflandırılmak yerine bu isteği tamamlamasını sağlıyor (ADR-008).
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
    # Slot-doldurma cevabıysa ("4321"), kaydedilecek `reason` orijinal istek olmalı.
    carried = state.get("carried_pending_request")
    reason_source = carried.original_message if carried else state["user_query"]
    arguments = _build_arguments(tool_name, entity_value, reason_source)

    record = await tool_client.call_tool(tool_name, arguments)

    # Özetleyen LLM çağrısı başarısız olursa (sağlayıcı kesintisi) deterministik
    # formatlayıcıya düş — araç sonucu record'da zaten gerçek veri olarak duruyor.
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
    """Modelin uydurduğu, konuşmada hiç geçmemiş bir finansal kimlikle
    (account_id/card_last4) araç çalıştırmayı reddeder (ADR-009). Grounded
    değilse modele geri beslenecek hata string'i, aksi halde None döner.
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
    """`bind_tools` için LangChain araç nesnelerini kurar. Her çağrı önce
    argüman grounding kontrolünden geçer, sonucu (başarılı/reddedilmiş/başarısız
    fark etmeksizin) `collected`'a eklenir — API'nin `tool_calls` alanı
    döngünün gerçekte ne yaptığını yansıtsın diye.
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
        """IBAN'a göre bir DemoBank hesabının güncel bakiyesini ve para birimini getirir."""
        return await _call("get_balance", {"account_id": account_id})

    @lc_tool
    async def list_transactions(account_id: str, limit: int = 10) -> str:
        """Bir DemoBank hesabının en son işlemlerini IBAN'a göre listeler."""
        return await _call("list_transactions", {"account_id": account_id, "limit": limit})

    @lc_tool
    async def block_card(card_last4: str, reason: str) -> str:
        """Son 4 hanesiyle belirtilen bir DemoBank kartını bloklar (kayıp/çalıntı bildirimi vb.)."""
        return await _call("block_card", {"card_last4": card_last4, "reason": reason})

    @lc_tool
    async def open_support_ticket(subject: str, description: str) -> str:
        """Diğer araçların çözemediği durumlar için bir destek talebi açar."""
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
            break  # sağlayıcı hatası — şimdiye kadar toplananla devam et

        messages.append(ai_message)
        tool_calls = ai_message.tool_calls[:_MAX_TOOL_CALLS_PER_HOP]

        if not tool_calls:
            final_text = content_to_text(ai_message.content)
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
    """Araç istemcisi + LLM'i async bir LangGraph düğümüne bağlar. Deterministik
    mi yoksa LLM-planlı yola mı gideceği burada bir kere karara bağlanır."""
    use_reasoning = not is_fake_model(llm)

    async def tool_agent_node(state: GraphState) -> dict[str, object]:
        if use_reasoning:
            return await _reasoning_tool_call(state, tool_client, llm, settings)
        return await _deterministic_tool_call(state, tool_client, llm)

    return tool_agent_node
