"""API + modüller arası veri sözleşmeleri.

Bu kod tabanındaki her sınır (HTTP istek/yanıt, MCP araç payload'ları, bir
düğüm sınırını geçen LangGraph state alanları) `dict` olarak dolaştırılmak
yerine burada tiplendiriliyor. Diğer modüllerin paylaşılan şekilleri import
etmesi gereken tek yer burası — ajan kod tabanları büyüdükçe ortaya çıkan
"aynı kavram, beş farklı dict şekli" savrulmasını önlüyor.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class IntentLabel(StrEnum):
    RAG_QUERY = "RAG_QUERY"  # bilgi tabanından yanıtlanabilecek politika/SSS sorusu
    ACCOUNT_ACTION = "ACCOUNT_ACTION"  # bakiye/ekstre/hesap bilgisi sorgusu
    TRANSACTION_ACTION = "TRANSACTION_ACTION"  # transfer, işlem geçmişi
    CARD_ACTION = "CARD_ACTION"  # blokla/kaldır/limit değişikliği
    SMALL_TALK = "SMALL_TALK"
    ESCALATE = "ESCALATE"  # kullanıcı açıkça bir insan istiyor
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class GuardrailFlag(StrEnum):
    """`guardrail_agent.py`'nin sonuçları için kapalı bir kelime dağarcığı.

    `IntentLabel` ve `EntityType` ile aynı sebepten bir enum olarak modellendi
    (uydurma string'ler değil): bir string flag'deki bir yazım hatası, kontrol
    edildiği hiçbir yerde sessizce eşleşmez, ne mypy ne de bir test bunu yakalar.
    """

    PII_REDACTED = "PII_REDACTED"
    FINANCIAL_ADVICE_BLOCKED = "FINANCIAL_ADVICE_BLOCKED"
    ESCALATED_ITERATION_LIMIT = "ESCALATED_ITERATION_LIMIT"
    NO_DRAFT_PRODUCED = "NO_DRAFT_PRODUCED"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
    MODEL_IDENTITY_REDACTED = "MODEL_IDENTITY_REDACTED"


class EntityType(StrEnum):
    IBAN = "IBAN"
    AMOUNT = "AMOUNT"
    CURRENCY = "CURRENCY"
    DATE = "DATE"
    CARD_LAST4 = "CARD_LAST4"
    ACCOUNT_TYPE = "ACCOUNT_TYPE"
    PERSON_NAME = "PERSON_NAME"


class Entity(BaseModel):
    type: EntityType
    value: str = Field(description="Kullanıcının mesajında geçtiği haliyle ham alt-dize")
    normalized: str | None = Field(
        default=None, description="Kanonikleştirilmiş değer, ör. boşluksuz bir IBAN"
    )
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class Citation(BaseModel):
    doc_id: str
    title: str
    source: str
    snippet: str
    score: float = Field(ge=0.0, le=1.0)


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: dict[str, object]
    result: dict[str, object] | str | None = None
    ok: bool
    error: str | None = None
    latency_ms: float = Field(ge=0.0)


class AgentTraceStep(BaseModel):
    node: str
    summary: str
    metadata: dict[str, object] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PendingEntityRequest(BaseModel):
    """`tool_agent`, eksik bir varlık yüzünden bir turn'ü kısa devre yaptırdığında
    (ör. kartın son 4 hanesini sorduğunda) set edilir ve `memory_agent`
    tarafından kalıcı hale getirilir ki *bir sonraki* turn genelde sadece
    "1234", ner_extractor'ın normalde gerektireceği bir anahtar kelime olmadan
    — sıfırdan yeniden sınıflandırılmak yerine bu isteği tamamlıyor olarak
    anlaşılsın.  ADR-008.

    `original_message`, asıl *neden*i açıklayan isteği tutuyor ("kartımı
    blokla, çalındı")  bu olmasaydı, tamamlanmış bir slot-doldurma çıplak
    takip cevabını ("4321") `block_card`'ın `reason` argümanı olarak
    kullanırdı — doğru ama kullanıcıya geri okununca anlamsız.
    """

    intent: IntentLabel
    entity_type: EntityType
    original_message: str


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None, description="Yeni bir konuşma başlatmak için boş bırakın"
    )
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    intent: IntentLabel
    entities: list[Entity] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    trace: list[AgentTraceStep] = Field(default_factory=list)
    guardrail_flags: list[GuardrailFlag] = Field(default_factory=list)
    iterations: int = Field(ge=0)


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ErrorCode(StrEnum):
    """Sadece HTTP seviyesindeki hata kodları.

    Ajan-seviyesi "hatalar" (iterasyon limitine ulaşma, guardrail'in bir
    cevabı engellemesi, boş dönen bir araç çağrısı) bu enum'da yok — ADR-006'daki
    "beklenen sonuçlar için raise etme, bir sonuç döndür" prensibine göre,
    `GuardrailFlag` taşıyan başarılı `ChatResponse`'lar olarak modelleniyorlar.
    Burada listelenen her üye kodda gerçekten bir yerde raise ediliyor —
    aksi halde ilk hata ayıklamada "bu zaten ele alınmış" yanılsaması yaratır.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, object] | None = None
