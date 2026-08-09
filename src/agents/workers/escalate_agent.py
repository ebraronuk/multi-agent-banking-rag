"""IntentLabel.ESCALATE ve IntentLabel.OUT_OF_SCOPE'u işler.

Bilinçli olarak LLM'siz (bkz. ADR-013) — bir insana aktarım, tek bir statik
mesaj değil, script'li bir akış: aktarım+temsilci karşılaması+kimlik
doğrulama isteği tek bir turda (`verifying`) -> doğrulama başarılı, sorunu
sor (`awaiting_issue`) -> sorunu kaydet, somut bir süre ver (`resolved`) ->
kapanış (`None`'a döner). Aşama `agents/memory.py` üzerinden turlar arası
taşınıyor (`carried_escalation_stage` bu turun girdisi, `escalation_stage`
çıktısı).

Aktarım + karşılama + doğrulama isteği tek turda birleşik: ayrı bir
"merhaba, nasıl yardımcı olabilirim" turu beklemek hem gereksiz bir
round-trip hem de kullanıcının hemen ardından yazdığı mesajı (genelde
şikayetin kendisi) görmezden bırakıyordu. Sorun da sadece bir kez,
doğrulamadan sonra soruluyor — gerçek bir banka desteğinin de yaptığı gibi
önce kimlik, sonra konu.

`escalation_stage` aktifken bu düğümün döndürdüğü `intent` her zaman
`ESCALATE` — o turun ham sınıflandırması API'nin `intent` alanına sızmıyor
(aksi halde Aylin'in mesajının altında yanlış bir etiket görünebiliyordu).
`supervisor.py` da script aktifken sınıflandırmaya bakmadan doğrudan buraya
yönlendiriyor — bir LLM'in "aktarım yapıldı mı" gibi bir mesajı yanlış
sınıflandırıp (ör. "aktarım" kelimesinin iki anlamı yüzünden
TRANSACTION_ACTION sanıp) akışı atlaması artık mümkün değil.

LLM'siz olmasının sebebi bu yüzden çift: modelin bankanın tutamayacağı bir
vaadi doğaçlaması riski (ADR-006/ADR-009 ile aynı ilke), ve yukarıdaki
yanlış-sınıflandırma riski. Gerçek bir insan hiçbir aşamada bağlanmıyor —
"Aylin" script'li, sabit bir persona; bu bir portföy demosu, gerçek bir
müşteri hizmetleri kuyruğu yok.
"""

from __future__ import annotations

import re

from agents.state import GraphState
from schemas.dto import AgentTraceStep, IntentLabel

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

# Gerçek bir hesap sorgusu değil (bkz. modül docstring'i) — bir NER entity'si
# ya da banking_repository'ye karşı bir doğrulama değil, sadece "kullanıcı 4
# haneli bir şey yazdı mı" kontrolü. Demo modunda herhangi bir 4 haneli
# numara kabul ediliyor (arayüzde de belirtiliyor).
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

    return {
        "draft_answer": _OUT_OF_SCOPE_MESSAGE,
        "escalation_stage": None,
        "trace": [AgentTraceStep(node="escalate", summary=f"out of scope for intent={intent}")],
    }


def _step(message: str, next_stage: str | None, summary: str) -> dict[str, object]:
    return {
        "draft_answer": message,
        "escalation_stage": next_stage,
        # Script aktifken raporlanan intent her zaman ESCALATE — o turun ham
        # sınıflandırması ne olursa olsun (bkz. modül docstring'i).
        "intent": IntentLabel.ESCALATE,
        "trace": [AgentTraceStep(node="escalate", summary=summary)],
    }
