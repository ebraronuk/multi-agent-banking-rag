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


def advance_intent_node(state: GraphState) -> dict[str, object]:
    """Kuyruktaki bir sonraki niyeti aktif yapar, bu pass'in taslağını sentez
    listesine ekler. LLM'siz, saf bir state geçişi — supervisor_node gibi.
    """
    extra = list(state.get("extra_intents", []))
    next_intent = extra.pop(0)
    finished_draft = state.get("draft_answer")

    return {
        "intent": next_intent,
        "extra_intents": extra,
        "collected_drafts": [finished_draft] if finished_draft else [],
        "draft_answer": None,
        "worker_pass_done": False,
        "tool_agent_done": False,
        "trace": [
            AgentTraceStep(node=NODE_ADVANCE_INTENT, summary=f"advancing to extra intent={next_intent}")
        ],
    }


def build_supervisor_router(settings: Settings) -> Callable[[GraphState], str]:
    """Bu run'ın iterasyon sınırına bağlı koşullu-kenar fonksiyonunu döner."""

    def route(state: GraphState) -> str:
        intent = state.get("intent")
        iteration_count = state.get("iteration_count", 0)

        # Konuşma zaten insana aktarım akışının ortasındaysa (bkz. ADR-013),
        # sınıflandırıcıya güvenmiyoruz. Somut kanıt: canlıda "aktarım yapıldı
        # mı" gerçek bir LLM tarafından TRANSACTION_ACTION sanıldı ("aktarım"
        # kelimesinin iki anlamı yüzünden) ve akış tamamen atlanıp kendi
        # kafasından bir yanıt uyduruldu.
        #
        # TEK istisna: doğrulama bittikten sonra (`awaiting_issue`) kullanıcı
        # gerçekten cevaplanabilir bir bilgi sorusu sorarsa (RAG_QUERY),
        # `escalate_node` (LLM'siz, bkz. ADR-006/ADR-013) bunu cevaplayamaz —
        # tek yapabildiği "bu konuyu kaydettim" demek, ki bu da canlıda
        # görüldü: "EFT limitim ne kadar" gibi anında cevaplanabilecek bir
        # soruya anlamsızca "kaydettim, 24 saat içinde dönüş yapacağız"
        # cevabı veriyordu. `rag_agent` gerçekten cevaplayabiliyorsa
        # bekletmenin bir anlamı yok. `verifying` aşamasında bu istisna
        # yok — kimlik doğrulanmadan hiçbir soru script'i atlayamıyor.
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
