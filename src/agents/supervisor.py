"""Supervisor: grafiğin rota kararını veren kısmı.

Bilinçli olarak bir LLM çağrısı değil — birkaç IntentLabel karşılaştırması
ve bir iterasyon sınırından ibaret. Yukarı akıştaki sınıflandırma kararı
zaten belirlediği için burada bir modele sormak sadece gecikme, maliyet ve
yeni bir hata türü (yanlış yönlendirme) eklerdi. Yönlendirdiği worker'lar
(rag_agent, tool_agent...) gerçek akıl yürütme gerektiren kısımlarda LLM'i
zaten kullanıyor. Bkz. docs/decisions/ADR-002-supervisor-routing.md.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts.sub_query_prompt import SUB_QUERY_ISOLATION_SYSTEM_PROMPT
from agents.state import GraphState
from app.core.config import Settings
from app.core.llm import is_fake_model, safe_ainvoke
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
NODE_ADVANCE_INTENT = "advance_intent"
NODE_SYNTHESIZER = "synthesizer"


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


def _finish_only(state: GraphState) -> str:
    """Zincire devam etme — kalan ek niyetler varsa bile bırak, sadece topla ya da bitir.

    İterasyon limiti tam bu turda dolduysa kullanılıyor: yeni bir niyete
    geçmek (advance) sınırın amacını boşa çıkarırdı.
    """
    if state.get("collected_drafts"):
        return NODE_SYNTHESIZER
    return NODE_GUARDRAIL


def _advance_or_finish(state: GraphState) -> str:
    """Aktif niyetin worker'ı işini bitirdi — kuyrukta başka niyet var mı diye bakar."""
    if state.get("extra_intents"):
        return NODE_ADVANCE_INTENT
    return _finish_only(state)


async def _isolate_sub_query(full_text: str, target_intent: IntentLabel, llm: BaseChatModel) -> str:
    """`full_text`'in yalnızca `target_intent`'le ilgili kısmını döner.

    Sadece gerçek modda çağrılır (bkz. çağıran); başarısız olursa da tam
    metne düşer — rag_agent/synthesizer'daki aynı fail-open desen.
    """
    isolated = await safe_ainvoke(
        llm,
        [
            SystemMessage(content=SUB_QUERY_ISOLATION_SYSTEM_PROMPT),
            HumanMessage(content=f"Konu: {target_intent}\n\nMesaj: {full_text}"),
        ],
        node=NODE_ADVANCE_INTENT,
    )
    return isolated or full_text


def build_advance_intent_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    """Kuyruktaki bir sonraki niyeti aktif yapar, bu pass'in taslağını sentez
    listesine ekler.

    RAG_QUERY'ye geçerken (gerçek modda) mesajın sadece o kısmını izole edip
    `active_sub_query`'ye yazar — aksi halde rag_agent bileşik mesajın tamamını
    arıyor, retrieval kalitesi düşüyor (ADR-012'nin bilinen sınırıydı, artık
    RAG_QUERY için düzeltildi). Diğer niyetler tam mesajı almaya devam ediyor;
    onlar entity-grounded çalıştığı için gürültüden aynı ölçüde etkilenmiyor.
    """

    async def advance_intent_node(state: GraphState) -> dict[str, object]:
        extra = list(state.get("extra_intents", []))
        next_intent = extra.pop(0)
        finished_draft = state.get("draft_answer")

        active_sub_query = None
        if next_intent == IntentLabel.RAG_QUERY and not is_fake_model(llm):
            active_sub_query = await _isolate_sub_query(state["user_query"], next_intent, llm)

        return {
            "intent": next_intent,
            "extra_intents": extra,
            "collected_drafts": [finished_draft] if finished_draft else [],
            "draft_answer": None,
            "worker_pass_done": False,
            "tool_agent_done": False,
            "active_sub_query": active_sub_query,
            "trace": [
                AgentTraceStep(
                    node=NODE_ADVANCE_INTENT, summary=f"advancing to extra intent={next_intent}"
                )
            ],
        }

    return advance_intent_node


def build_supervisor_router(settings: Settings) -> Callable[[GraphState], str]:
    """Bu run'ın iterasyon sınırına bağlı koşullu-kenar fonksiyonunu döner."""

    def route(state: GraphState) -> str:
        intent = state.get("intent")
        iteration_count = state.get("iteration_count", 0)

        # Aktarım script'i (ADR-013) aktifken sınıflandırıcıya güvenilmiyor —
        # yanlış sınıflandırma script'i atlayıp uydurma bir yanıta yol açabiliyordu.
        # Tek istisna: doğrulama sonrası (`awaiting_issue`) gerçekten
        # cevaplanabilir bir RAG_QUERY varsa, escalate_node (LLM'siz) onu
        # cevaplayamaz — rag_agent'a bırakılıyor. `verifying`'de istisna yok.
        stage = state.get("carried_escalation_stage")
        rag_can_answer_instead = stage == "awaiting_issue" and intent == IntentLabel.RAG_QUERY
        if stage is not None and not rag_can_answer_instead:
            return NODE_ESCALATE

        if intent in TOOL_DRIVEN_INTENTS:
            if iteration_count >= settings.max_agent_iterations:
                logger.warning(
                    "agent_iteration_limit_hit", intent=intent, iteration_count=iteration_count
                )
                return _finish_only(state)
            if not state.get("tool_agent_done"):
                return NODE_TOOL_AGENT
            return _advance_or_finish(state)

        if intent == IntentLabel.RAG_QUERY:
            if not state.get("worker_pass_done"):
                return NODE_RAG_AGENT
            return _advance_or_finish(state)

        if intent == IntentLabel.SMALL_TALK:
            if not state.get("worker_pass_done"):
                return NODE_SMALLTALK
            return _advance_or_finish(state)

        # ESCALATE, OUT_OF_SCOPE ya da sınıflandırılamamış/None bir intent —
        # hepsi tahmin etmek yerine insana aktarım mesajına düşüyor, zincire dahil değil.
        return NODE_ESCALATE

    return route
