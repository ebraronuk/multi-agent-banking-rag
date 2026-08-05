"""API + cross-module data contracts.

Every boundary in this codebase (HTTP request/response, MCP tool payloads,
LangGraph state fields that cross a node boundary) is typed here rather than
passed around as `dict`. This is the one place other modules should import
shared shapes from — avoids the "same concept, five different dict shapes"
drift that shows up in agent codebases as they grow.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class IntentLabel(StrEnum):
    RAG_QUERY = "RAG_QUERY"  # policy/FAQ question answerable from the knowledge base
    ACCOUNT_ACTION = "ACCOUNT_ACTION"  # balance/statement/account-info lookups
    TRANSACTION_ACTION = "TRANSACTION_ACTION"  # transfers, transaction history
    CARD_ACTION = "CARD_ACTION"  # block/unblock/limit change
    SMALL_TALK = "SMALL_TALK"
    ESCALATE = "ESCALATE"  # user explicitly wants a human
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class GuardrailFlag(StrEnum):
    """Closed vocabulary for `guardrail_agent.py`'s outcomes.

    Modeled as an enum (not ad-hoc strings) for the same reason `IntentLabel`
    and `EntityType` are: a typo in a bare string flag would silently fail to
    match anywhere it's checked, and neither mypy nor a test would catch it.
    """

    PII_REDACTED = "PII_REDACTED"
    FINANCIAL_ADVICE_BLOCKED = "FINANCIAL_ADVICE_BLOCKED"
    ESCALATED_ITERATION_LIMIT = "ESCALATED_ITERATION_LIMIT"
    NO_DRAFT_PRODUCED = "NO_DRAFT_PRODUCED"


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
    value: str = Field(description="Raw substring as it appeared in the user message")
    normalized: str | None = Field(
        default=None, description="Canonicalized value, e.g. IBAN without spaces"
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


class ChatRequest(BaseModel):
    conversation_id: str | None = Field(
        default=None, description="Omit to start a new conversation"
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
    """HTTP-level failure codes only.

    Agent-level "failures" (iteration limit hit, guardrail blocked a reply,
    a tool call came back empty) are deliberately NOT in this enum — they are
    modeled as successful `ChatResponse`s carrying `GuardrailFlag`s, per
    ADR-006 and the "return a result, don't raise for expected outcomes"
    principle. An enum member that never gets raised is worse than no member
    at all: it's a claim about behavior the code doesn't keep.
    """

    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ErrorResponse(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, object] | None = None
