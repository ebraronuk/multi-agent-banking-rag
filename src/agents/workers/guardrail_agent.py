"""Bir yanıt grafikten çıkmadan önceki son güvenlik geçişi.

Her yolda çalışır (RAG, araç, sohbet, aktarım) — hiçbir turn'ün atlayamadığı
tek düğüm bu, onu "sadece son adım" değil bir guardrail yapan da bu. Bilinçli
olarak saf, senkron ve LLM'siz: politika uygulaması, kontrol etmesi gereken
aynı girdiyle jailbreak edilebilecek bir modele bağımlı olmamalı.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agents.prompts.guardrail_prompt import (
    FINANCIAL_ADVICE_DISCLAIMER,
    ITERATION_LIMIT_MESSAGE,
    MODEL_IDENTITY_MESSAGE,
    NO_DRAFT_FALLBACK_MESSAGE,
    PROMPT_INJECTION_MESSAGE,
)
from agents.state import GraphState
from app.core.config import Settings
from app.core.logging import get_logger
from nlp.text_utils import turkish_lower
from schemas.dto import AgentTraceStep, GuardrailFlag

logger = get_logger(__name__)

# TR IBAN: TR + 2 kontrol hanesi + 5 banka kodu + 1 rezerv + 16 hesap = 26 karakter.
_IBAN_RE = re.compile(r"\bTR\d{2}(?:[ ]?\d{4}){5}[ ]?\d{2}\b", re.IGNORECASE)
# IBAN olmayan başka her uzun rakam dizisi (kart no, telefon benzeri) de aynı muameleyi görür.
_LONG_DIGIT_RUN_RE = re.compile(r"\b\d{6,}\b")

_ADVICE_KEYWORDS = (
    "hisse al",
    "kripto al",
    "şunu yatırım yap",
    "garanti getiri",
    "kesin kazanç",
)

# Kullanıcının mesajında aranıyor (LLM çıktısında değil), kural tabanlı bir katman.
# "kurallarını unut" gibi öneksiz kalıplar bilerek yok — "Müşteri temsilciniz
# kurallarını unutarak..." gibi gerçek şikayet cümleleriyle false-positive veriyordu.
_INJECTION_KEYWORDS = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard the above",
    "disregard previous instructions",
    "önceki talimatları yok say",
    "önceki talimatları unut",
    "sistem promptunu göster",
    "sistem promptunu yazdır",
    "system prompt'unu göster",
    "you are now",
    "developer mode",
    "dan modu",
    "jailbreak",
)

# LLM'in kendi kimliğini açık etmesi — banka asistanı kimliğinin dışına
# çıkıp hangi sağlayıcı/modelle çalıştığını söylemesi hem gereksiz bir bilgi
# sızıntısı hem de marka tutarlılığı sorunu.
_MODEL_IDENTITY_KEYWORDS = (
    "google gemini",
    "gemini modeli",
    "openai",
    "gpt-4",
    "gpt-3",
    "chatgpt",
    "anthropic",
    "claude modeli",
    "bir dil modeliyim",
    "bir yapay zeka modeliyim",
    "large language model",
    "büyük dil modeliyim",
)


def _redact_sensitive_numbers(text: str) -> tuple[str, bool]:
    redacted = False

    def _mask(match: re.Match[str]) -> str:
        nonlocal redacted
        raw = match.group(0)
        digits_only = re.sub(r"\s", "", raw)
        if len(digits_only) <= 4:
            return raw
        redacted = True
        return f"***{digits_only[-4:]}"

    text = _IBAN_RE.sub(_mask, text)
    text = _LONG_DIGIT_RUN_RE.sub(_mask, text)
    return text, redacted


def _contains_advice_language(text: str) -> bool:
    lowered = turkish_lower(text)
    return any(keyword in lowered for keyword in _ADVICE_KEYWORDS)


def _contains_injection_attempt(text: str) -> bool:
    lowered = turkish_lower(text)
    return any(keyword in lowered for keyword in _INJECTION_KEYWORDS)


def _reveals_model_identity(text: str) -> bool:
    lowered = turkish_lower(text)
    return any(keyword in lowered for keyword in _MODEL_IDENTITY_KEYWORDS)


def build_guardrail_node(settings: Settings) -> Callable[[GraphState], dict[str, object]]:
    def guardrail_node(state: GraphState) -> dict[str, object]:
        flags: list[GuardrailFlag] = list(state.get("guardrail_flags", []))
        draft = state.get("draft_answer")
        iteration_count = state.get("iteration_count", 0)
        tool_agent_pending = state.get("intent") is not None and not state.get(
            "tool_agent_done", True
        )

        if iteration_count >= settings.max_agent_iterations and tool_agent_pending:
            final_answer = ITERATION_LIMIT_MESSAGE
            flags.append(GuardrailFlag.ESCALATED_ITERATION_LIMIT)
        elif _contains_injection_attempt(state.get("user_query", "")):
            # Kullanıcının mesajına bakıyor, draft'a değil — LLM'e hiç
            # ulaşmadan reddediyoruz, modelin girdiyi nasıl yorumladığına
            # güvenmeden.
            final_answer = PROMPT_INJECTION_MESSAGE
            flags.append(GuardrailFlag.PROMPT_INJECTION_DETECTED)
        elif not draft:
            final_answer = NO_DRAFT_FALLBACK_MESSAGE
            flags.append(GuardrailFlag.NO_DRAFT_PRODUCED)
        elif _contains_advice_language(draft):
            final_answer = FINANCIAL_ADVICE_DISCLAIMER
            flags.append(GuardrailFlag.FINANCIAL_ADVICE_BLOCKED)
        elif _reveals_model_identity(draft):
            final_answer = MODEL_IDENTITY_MESSAGE
            flags.append(GuardrailFlag.MODEL_IDENTITY_REDACTED)
        else:
            final_answer, was_redacted = _redact_sensitive_numbers(draft)
            if was_redacted:
                flags.append(GuardrailFlag.PII_REDACTED)

        if flags:
            logger.info(
                "guardrail_flags_raised", flags=flags, conversation_id=state.get("conversation_id")
            )

        return {
            "final_answer": final_answer,
            "guardrail_flags": flags,
            "trace": [
                AgentTraceStep(
                    node="guardrail",
                    summary=f"guardrail resolved response ({len(flags)} flag(s))",
                    metadata={"flags": flags},
                )
            ],
        }

    return guardrail_node
