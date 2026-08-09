from __future__ import annotations

from agents.state import new_state
from agents.workers.escalate_agent import escalate_node
from schemas.dto import IntentLabel


def test_escalate_intent_combines_handoff_greeting_and_verification_request() -> None:
    # Regresyon: eskiden aktarım + temsilci karşılaması ayrı turlara
    # bölünmüştü, kullanıcı şikayetini iki kez yazmak zorunda kalıyordu
    # (canlıda görüldü). Artık tek turda hepsi birleşik.
    state = new_state("c1", "bir insanla konuşmak istiyorum")
    state["intent"] = IntentLabel.ESCALATE

    result = escalate_node(state)

    answer = result["draft_answer"].lower()
    assert "aktarıyorum" in answer
    assert "aylin" in answer
    assert "doğrula" in answer
    assert result["escalation_stage"] == "verifying"
    assert result["intent"] == IntentLabel.ESCALATE


def test_out_of_scope_gets_scope_message_listing_real_capabilities() -> None:
    state = new_state("c1", "yarın hava nasıl olacak")
    state["intent"] = IntentLabel.OUT_OF_SCOPE

    result = escalate_node(state)

    answer = result["draft_answer"].lower()
    assert "yardımcı olabilirim" in answer
    # Uydurma bir yetenek değil, gerçekten var olan bir tool_agent aracı:
    assert "bakiye" in answer
    assert result["escalation_stage"] is None


def test_verifying_stage_with_four_digit_code_asks_for_the_issue() -> None:
    state = new_state("c1", "4321")
    state["intent"] = IntentLabel.OUT_OF_SCOPE  # o turun ham sınıflandırması önemsiz
    state["carried_escalation_stage"] = "verifying"

    result = escalate_node(state)

    assert "doğruladım" in result["draft_answer"].lower()
    assert result["escalation_stage"] == "awaiting_issue"
    assert result["intent"] == IntentLabel.ESCALATE


def test_verifying_stage_without_four_digit_code_retries() -> None:
    state = new_state("c1", "hatırlamıyorum")
    state["carried_escalation_stage"] = "verifying"

    result = escalate_node(state)

    assert result["escalation_stage"] == "verifying"


def test_verifying_stage_extracts_code_embedded_in_a_sentence() -> None:
    state = new_state("c1", "müşteri numaram 4321 sanırım")
    state["carried_escalation_stage"] = "verifying"

    result = escalate_node(state)

    assert result["escalation_stage"] == "awaiting_issue"


def test_awaiting_issue_stage_acknowledges_and_gives_a_concrete_sla() -> None:
    # Şikayet SADECE burada, doğrulamadan sonra, tek seferde soruluyor.
    state = new_state("c1", "kartımdan izinsiz para çekilmiş")
    state["carried_escalation_stage"] = "awaiting_issue"

    result = escalate_node(state)

    answer = result["draft_answer"].lower()
    assert "24 saat" in answer
    assert result["escalation_stage"] == "resolved"


def test_resolved_stage_answers_timing_follow_up_instead_of_falling_out_of_script() -> None:
    # Regresyon: canlıda "ne zaman dönüş yapacaksınız" script'ten çıkıp genel
    # "kapsam dışı" cevabına düşüyordu — doğal bir takip sorusuna verilecek
    # en kötü cevap.
    state = new_state("c1", "ne zaman dönüş yapacaksınız")
    state["carried_escalation_stage"] = "resolved"

    result = escalate_node(state)

    assert "24 saat" in result["draft_answer"]
    assert result["escalation_stage"] is None


def test_resolved_stage_closes_gracefully_on_anything_else() -> None:
    state = new_state("c1", "teşekkürler")
    state["carried_escalation_stage"] = "resolved"

    result = escalate_node(state)

    assert result["escalation_stage"] is None
    assert result["draft_answer"]
