from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.state import new_state
from agents.supervisor import (
    NODE_ADVANCE_INTENT,
    NODE_ESCALATE,
    NODE_GUARDRAIL,
    NODE_RAG_AGENT,
    NODE_SMALLTALK,
    NODE_SYNTHESIZER,
    NODE_TOOL_AGENT,
    build_advance_intent_node,
    build_supervisor_router,
)
from app.core.config import Settings
from app.core.llm import FakeChatModel
from schemas.dto import IntentLabel


class _StubIsolationModel:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def ainvoke(self, messages: object) -> AIMessage:
        return AIMessage(content=self._response_text)


def _settings(max_iterations: int = 6) -> Settings:
    return Settings(max_agent_iterations=max_iterations)


def test_routes_rag_query_to_rag_agent() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limiti nedir?")
    state["intent"] = IntentLabel.RAG_QUERY

    assert route(state) == NODE_RAG_AGENT


def test_routes_small_talk_to_smalltalk() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "merhaba")
    state["intent"] = IntentLabel.SMALL_TALK

    assert route(state) == NODE_SMALLTALK


def test_carried_escalation_stage_overrides_intent_classification() -> None:
    # Regresyon: canlıda "aktarım yapıldı mı" gerçek bir LLM tarafından
    # TRANSACTION_ACTION olarak sınıflandırıldı ("aktarım" kelimesinin iki
    # anlamı yüzünden) ve escalate script'ini (bkz. ADR-013) tamamen atladı.
    # Script bir kez başladıysa (carried_escalation_stage None değilse) o
    # turda sınıflandırıcı ne derse desin escalate'e gidilmeli — script
    # doğrulama tamamlanana kadar kesintisiz sürüyor.
    route = build_supervisor_router(_settings())
    state = new_state("c1", "aktarım yapıldı mı")
    state["intent"] = IntentLabel.TRANSACTION_ACTION
    state["carried_escalation_stage"] = "verifying"

    assert route(state) == NODE_ESCALATE


def test_no_carried_escalation_stage_uses_normal_routing() -> None:
    # Script bitip carried_escalation_stage None'a döndüğünde (doğrulama
    # tamamlandığında) konuşma normal akışa geri dönüyor.
    route = build_supervisor_router(_settings())
    state = new_state("c1", "bakiyem ne kadar")
    state["intent"] = IntentLabel.ACCOUNT_ACTION
    state["carried_escalation_stage"] = None

    assert route(state) == NODE_TOOL_AGENT


def test_verifying_stage_forces_escalate_even_for_rag_query() -> None:
    # Kimlik doğrulanmadan hiçbir soru script'i atlayamaz — awaiting_issue'daki
    # RAG_QUERY istisnası (aşağıya bak) burada geçerli değil.
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limitiniz ne kadar?")
    state["intent"] = IntentLabel.RAG_QUERY
    state["carried_escalation_stage"] = "verifying"

    assert route(state) == NODE_ESCALATE


def test_awaiting_issue_stage_lets_a_real_rag_query_through() -> None:
    # Regresyon: canlıda "awaiting_issue" aşamasında "EFT limitim ne kadar"
    # gibi anında cevaplanabilecek bir soru bile "bu konuyu kaydettim, 24 saat
    # içinde dönüş yapacağız" cevabını alıyordu — escalate_node LLM'siz
    # olduğu için bunu gerçekten cevaplayamıyor, rag_agent cevaplayabiliyor.
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limitim ne kadar?")
    state["intent"] = IntentLabel.RAG_QUERY
    state["carried_escalation_stage"] = "awaiting_issue"
    state["worker_pass_done"] = False

    assert route(state) == NODE_RAG_AGENT


def test_awaiting_issue_stage_rag_query_finishes_normally_after_answering() -> None:
    # rag_agent cevapladıktan sonra (worker_pass_done=True) sonsuz döngüye
    # girmemeli, escalate_node'a da geri dönmemeli — normal RAG_QUERY
    # akışındaki gibi guardrail'e düşmeli.
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limitim ne kadar?")
    state["intent"] = IntentLabel.RAG_QUERY
    state["carried_escalation_stage"] = "awaiting_issue"
    state["worker_pass_done"] = True

    assert route(state) == NODE_GUARDRAIL


def test_awaiting_issue_stage_still_escalates_non_rag_intents() -> None:
    # İstisna sadece RAG_QUERY için — ör. bir CARD_ACTION denemesi hâlâ
    # escalate_node'a gidip "bu konuyu kaydettim" akışına giriyor.
    route = build_supervisor_router(_settings())
    state = new_state("c1", "kartımdan izinsiz para çekilmiş")
    state["intent"] = IntentLabel.CARD_ACTION
    state["carried_escalation_stage"] = "awaiting_issue"

    assert route(state) == NODE_ESCALATE


def test_routes_unclassified_and_out_of_scope_to_escalate() -> None:
    route = build_supervisor_router(_settings())

    state = new_state("c1", "?")
    state["intent"] = None
    assert route(state) == NODE_ESCALATE

    state["intent"] = IntentLabel.OUT_OF_SCOPE
    assert route(state) == NODE_ESCALATE

    state["intent"] = IntentLabel.ESCALATE
    assert route(state) == NODE_ESCALATE


def test_tool_driven_intent_loops_until_done() -> None:
    route = build_supervisor_router(_settings(max_iterations=6))
    state = new_state("c1", "bakiyem ne kadar")
    state["intent"] = IntentLabel.ACCOUNT_ACTION
    state["tool_agent_done"] = False
    state["iteration_count"] = 1

    assert route(state) == NODE_TOOL_AGENT

    state["tool_agent_done"] = True
    assert route(state) == NODE_GUARDRAIL


def test_tool_driven_intent_stops_at_iteration_cap_even_if_not_done() -> None:
    route = build_supervisor_router(_settings(max_iterations=3))
    state = new_state("c1", "kartımı blokla")
    state["intent"] = IntentLabel.CARD_ACTION
    state["tool_agent_done"] = False
    state["iteration_count"] = 3

    assert route(state) == NODE_GUARDRAIL


def test_rag_query_second_pass_goes_to_guardrail_when_no_extra_intent() -> None:
    # rag_agent kendi işini bitirip supervisor'a döndüğünde (worker_pass_done),
    # kuyrukta bekleyen bir niyet yoksa bugünküyle birebir aynı davranış.
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limitiniz ne kadar?")
    state["intent"] = IntentLabel.RAG_QUERY
    state["worker_pass_done"] = True

    assert route(state) == NODE_GUARDRAIL


def test_rag_query_second_pass_advances_when_extra_intent_queued() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "EFT limitiniz ne kadar ve merhaba")
    state["intent"] = IntentLabel.RAG_QUERY
    state["worker_pass_done"] = True
    state["extra_intents"] = [IntentLabel.SMALL_TALK]

    assert route(state) == NODE_ADVANCE_INTENT


def test_tool_driven_intent_advances_to_extra_intent_once_done() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["intent"] = IntentLabel.CARD_ACTION
    state["tool_agent_done"] = True
    state["extra_intents"] = [IntentLabel.RAG_QUERY]

    assert route(state) == NODE_ADVANCE_INTENT


def test_routes_to_synthesizer_once_queue_empty_and_drafts_collected() -> None:
    route = build_supervisor_router(_settings())
    state = new_state("c1", "merhaba")
    state["intent"] = IntentLabel.SMALL_TALK
    state["worker_pass_done"] = True
    state["extra_intents"] = []
    state["collected_drafts"] = ["ilk taslak"]

    assert route(state) == NODE_SYNTHESIZER


def test_iteration_cap_never_advances_to_extra_intent() -> None:
    # Limit dolduysa kalan kuyruk sessizce bırakılıyor — advance etmek
    # sınırın amacını boşa çıkarırdı (bkz. ADR-012'nin bilinen sınırları).
    route = build_supervisor_router(_settings(max_iterations=2))
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["intent"] = IntentLabel.CARD_ACTION
    state["tool_agent_done"] = False
    state["iteration_count"] = 2
    state["extra_intents"] = [IntentLabel.RAG_QUERY]

    assert route(state) == NODE_GUARDRAIL


async def test_advance_intent_node_pops_queue_and_resets_pass_flags() -> None:
    node = build_advance_intent_node(FakeChatModel())
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["extra_intents"] = [IntentLabel.RAG_QUERY, IntentLabel.SMALL_TALK]
    state["draft_answer"] = "kartınız bloklandı"
    state["worker_pass_done"] = True
    state["tool_agent_done"] = True

    result = await node(state)

    assert result["intent"] == IntentLabel.RAG_QUERY
    assert result["extra_intents"] == [IntentLabel.SMALL_TALK]
    assert result["collected_drafts"] == ["kartınız bloklandı"]
    assert result["draft_answer"] is None
    assert result["worker_pass_done"] is False
    assert result["tool_agent_done"] is False


async def test_advance_intent_node_does_not_collect_an_empty_draft() -> None:
    node = build_advance_intent_node(FakeChatModel())
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["extra_intents"] = [IntentLabel.RAG_QUERY]
    state["draft_answer"] = None

    result = await node(state)

    assert result["collected_drafts"] == []


async def test_advance_intent_node_fake_model_leaves_active_sub_query_unset() -> None:
    # Fake modda no-op: sub-query izolasyonu için gerçek bir model yok,
    # rag_agent tam mesajla çalışmaya devam ediyor (mevcut davranış).
    node = build_advance_intent_node(FakeChatModel())
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["extra_intents"] = [IntentLabel.RAG_QUERY]

    result = await node(state)

    assert result["active_sub_query"] is None


async def test_advance_intent_node_real_model_isolates_rag_sub_query() -> None:
    # Regresyon (ADR-012'nin bilinen sınırı): bileşik bir mesajın tamamı
    # RAG'e gidince retrieval kalitesi düşüyordu — artık gerçek modda sadece
    # ilgili kısım izole edilip active_sub_query'ye yazılıyor.
    stub = _StubIsolationModel("EFT limitiniz ne kadar")
    node = build_advance_intent_node(stub)  # type: ignore[arg-type]
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["extra_intents"] = [IntentLabel.RAG_QUERY]

    result = await node(state)

    assert result["active_sub_query"] == "EFT limitiniz ne kadar"


async def test_advance_intent_node_only_isolates_for_rag_query() -> None:
    # Diğer niyetler (SMALL_TALK, tool-driven) entity-grounded ya da düşük
    # riskli çalışıyor — izolasyon maliyeti sadece RAG_QUERY için gerekli.
    stub = _StubIsolationModel("bu hiç çağrılmamalı")
    node = build_advance_intent_node(stub)  # type: ignore[arg-type]
    state = new_state("c1", "kartımı blokla ve merhaba")
    state["extra_intents"] = [IntentLabel.SMALL_TALK]

    result = await node(state)

    assert result["active_sub_query"] is None
