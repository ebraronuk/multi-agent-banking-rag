"""Supervisor: grafiğin rota kararını veren kısmı.

Bilinçli olarak bir LLM çağrısı değil — birkaç IntentLabel karşılaştırması
ve bir iterasyon sınırından ibaret. Yukarı akıştaki sınıflandırma kararı
zaten belirlediği için burada bir modele sormak sadece gecikme, maliyet ve
yeni bir hata türü (yanlış yönlendirme) eklerdi. Yönlendirdiği worker'lar
(rag_agent, tool_agent...) gerçek akıl yürütme gerektiren kısımlarda LLM'i
zaten kullanıyor. Bkz. docs/decisions/ADR-002-supervisor-routing.md.
"""

from __future__ import annotations

from collections.abc import Callable

from agents.state import GraphState
from app.core.config import Settings
from app.core.logging import get_logger
from schemas.dto import AgentTraceStep, IntentLabel

logger = get_logger(__name__)

TOOL_DRIVEN_INTENTS = frozenset(
    {IntentLabel.ACCOUNT_ACTION, IntentLabel.TRANSACTION_ACTION, IntentLabel.CARD_ACTION}
)

NODE_RAG_AGENT = "rag_agent"
NODE_TOOL_AGENT = "tool_agent"
NODE_SMALLTALK = "smalltalk"
NODE_ESCALATE = "escalate"
NODE_GUARDRAIL = "guardrail"


def supervisor_node(state: GraphState) -> dict[str, object]:
    """Tek işi bir trace kaydı bırakmak olan, işlemsiz bir geçiş düğümü.

    Koşullu kenarlar doğrudan herhangi bir düğümden dallanabilir, ama dallanma
    noktasına kendi adını vermek grafiği (ve API'ye dönen trace'i) okunur
    kılıyor — "supervisor rotayı değerlendirdi" görünmez bir `if` değil,
    gerçek, izlenebilir bir adım.
    """
    intent = state.get("intent")
    return {
        "trace": [
            AgentTraceStep(
                node="supervisor",
                summary=f"routing decision for intent={intent}",
                metadata={"iteration_count": state.get("iteration_count", 0)},
            )
        ]
    }


def build_supervisor_router(settings: Settings) -> Callable[[GraphState], str]:
    """Bu run'ın iterasyon sınırına bağlı koşullu-kenar fonksiyonunu döner."""

    def route(state: GraphState) -> str:
        intent = state.get("intent")
        iteration_count = state.get("iteration_count", 0)

        if intent in TOOL_DRIVEN_INTENTS:
            if iteration_count >= settings.max_agent_iterations:
                logger.warning(
                    "agent_iteration_limit_hit", intent=intent, iteration_count=iteration_count
                )
                return NODE_GUARDRAIL
            if state.get("tool_agent_done"):
                return NODE_GUARDRAIL
            return NODE_TOOL_AGENT

        if intent == IntentLabel.RAG_QUERY:
            return NODE_RAG_AGENT

        if intent == IntentLabel.SMALL_TALK:
            return NODE_SMALLTALK

        # ESCALATE, OUT_OF_SCOPE ya da sınıflandırılamamış/None bir intent —
        # hepsi tahmin etmek yerine insana aktarım mesajına düşüyor.
        return NODE_ESCALATE

    return route
