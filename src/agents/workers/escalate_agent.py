"""IntentLabel.ESCALATE ve IntentLabel.OUT_OF_SCOPE'u işler.

LLM'siz, script'li bir akış: aktarım+temsilci karşılaması+doğrulama isteği
tek turda (`verifying`) -> sorunu al (`awaiting_issue`) -> kaydet + SLA ver
(`resolved`) -> kapanış. Aşama `agents/memory.py` üzerinden taşınıyor
(`carried_escalation_stage` girdi, `escalation_stage` çıktı). Gerekçe ve
tasarım geçmişi: ADR-013.
"""

from __future__ import annotations

import re

from agents.state import GraphState
from schemas.dto import AgentTraceStep, EntityType, IntentLabel

_AGENT_NAME = "Aylin"

_HANDOFF_AND_VERIFICATION_REQUEST = (
    f"Sizi bir müşteri temsilcisine aktarıyorum. Merhaba, ben müşteri temsilciniz "
    f"{_AGENT_NAME}, hemen ilgileniyorum. Öncelikle sizi doğrulamam gerekiyor — müşteri ya "
    f"da kart numaranızın son 4 hanesini paylaşır mısınız?"
)
_VERIFICATION_RETRY = (
    "Bunu bir doğrulama numarası olarak tanıyamadım — 4 haneli müşteri ya da kart "
    "numaranızı paylaşabilir misiniz?"
)
_ASK_FOR_ISSUE = (
    "Teşekkürler, kimliğinizi doğruladım. Şimdi size nasıl yardımcı olabilirim, sorununuzu "
    "kısaca anlatır mısınız?"
)
_ISSUE_ACKNOWLEDGED = (
    "Anladım, bu konuyu kaydettim. En geç 24 saat içinde bir uzmanımız sizinle iletişime "
    "geçecek. Başka bir konuda yardımcı olabilir miyim?"
)
_TIMING_REMINDER = "En geç 24 saat içinde bir uzmanımız sizinle iletişime geçecek."
_CLOSING = "Rica ederim, başka bir konuda da yardımcı olabilirim."
_OUT_OF_SCOPE_MESSAGE = (
    "Bu konuda size yardımcı olamıyorum. Ama hesap bakiyeniz, işlem geçmişiniz, kart "
    "işlemleriniz ya da havale/EFT limitleri ve hesap ücretleri gibi banka politikalarımız "
    "hakkında bir sorunuz varsa yardımcı olabilirim, ya da sizi bir müşteri temsilcisine "
    "yönlendirebilirim."
)

# Beklenen formatta olmayan bir slot-fill cevabı (ör. IBAN yerine "4321")
# için hedefli bir retry — genel _OUT_OF_SCOPE_MESSAGE'a düşmesin.
_PENDING_RETRY_MESSAGES: dict[EntityType, str] = {
    EntityType.IBAN: (
        "Bu bir IBAN gibi görünmüyor. TR ile başlayan 26 haneli hesap numaranızı tam "
        "olarak paylaşır mısınız?"
    ),
    EntityType.CARD_LAST4: (
        "Bu bir kart numarası gibi görünmüyor. Kartınızın son 4 hanesini rakamla yazar "
        "mısınız?"
    ),
}

# Gerçek bir doğrulama değil, sadece "4 haneli bir şey yazdı mı" kontrolü —
# demo modunda herhangi bir 4 haneli numara kabul ediliyor (arayüzde belirtiliyor).
_VERIFICATION_CODE_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")
_TIMING_FOLLOW_UP_RE = re.compile(r"ne zaman|kaç (gün|saat)|ne kadar sürer|süre", re.IGNORECASE)


def escalate_node(state: GraphState) -> dict[str, object]:
    intent = state.get("intent")
    stage = state.get("carried_escalation_stage")

    if stage == "verifying":
        if _VERIFICATION_CODE_RE.search(state["user_query"]):
            return _step(_ASK_FOR_ISSUE, "awaiting_issue", "verification succeeded")
        return _step(_VERIFICATION_RETRY, "verifying", "verification retry (no 4-digit code found)")

    if stage == "awaiting_issue":
        return _step(_ISSUE_ACKNOWLEDGED, "resolved", "issue logged, gave SLA")

    if stage == "resolved":
        if _TIMING_FOLLOW_UP_RE.search(state["user_query"]):
            return _step(_TIMING_REMINDER, None, "answered timing follow-up, closing")
        return _step(_CLOSING, None, "closing")

    if intent == IntentLabel.ESCALATE:
        return _step(
            _HANDOFF_AND_VERIFICATION_REQUEST,
            "verifying",
            "handed off, agent greeted, verification requested",
        )

    pending = state.get("carried_pending_request")
    if pending is not None and pending.entity_type in _PENDING_RETRY_MESSAGES:
        return {
            "draft_answer": _PENDING_RETRY_MESSAGES[pending.entity_type],
            "escalation_stage": None,
            # Bir sonraki turn'ün bare cevabını ADR-008'in slot-doldurma
            # yoluyla tanıyabilmesi için taşınıyor.
            "pending_entity_request": pending,
            "trace": [
                AgentTraceStep(
                    node="escalate",
                    summary=f"pending {pending.entity_type} not satisfied by this turn, retry prompt",
                )
            ],
        }

    return {
        "draft_answer": _OUT_OF_SCOPE_MESSAGE,
        "escalation_stage": None,
        "trace": [AgentTraceStep(node="escalate", summary=f"out of scope for intent={intent}")],
    }


def _step(message: str, next_stage: str | None, summary: str) -> dict[str, object]:
    return {
        "draft_answer": message,
        "escalation_stage": next_stage,
        "intent": IntentLabel.ESCALATE,  # script aktifken ham sınıflandırma sızmaz
        "trace": [AgentTraceStep(node="escalate", summary=summary)],
    }
