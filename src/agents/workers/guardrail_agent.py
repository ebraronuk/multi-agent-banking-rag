"""Final safety pass before a response leaves the graph.

Runs on every path (RAG, tool, small talk, escalate) — it is the one node no
turn can skip, which is what makes it a guardrail rather than just "the last
step". Pure, synchronous, and LLM-free on purpose: policy enforcement should
not itself depend on a model that can be jailbroken by the same input it's
supposed to be checking.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agents.prompts.guardrail_prompt import (
    FINANCIAL_ADVICE_DISCLAIMER,
    ITERATION_LIMIT_MESSAGE,
    NO_DRAFT_FALLBACK_MESSAGE,
)
from agents.state import GraphState
from app.core.config import Settings
from app.core.logging import get_logger
from schemas.dto import AgentTraceStep

logger = get_logger(__name__)

# TR IBAN: TR + 2 check digits + 5 bank code + 1 reserve + 16 account = 26 chars total.
_IBAN_RE = re.compile(r"\bTR\d{2}(?:[ ]?\d{4}){5}[ ]?\d{2}\b", re.IGNORECASE)
# Any other long digit run (PAN, phone-like) that isn't an IBAN gets the same treatment.
_LONG_DIGIT_RUN_RE = re.compile(r"\b\d{6,}\b")

_ADVICE_KEYWORDS = (
    "hisse al",
    "kripto al",
    "şunu yatırım yap",
    "garanti getiri",
    "kesin kazanç",
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
    lowered = text.lower()
    return any(keyword in lowered for keyword in _ADVICE_KEYWORDS)


def build_guardrail_node(settings: Settings) -> Callable[[GraphState], dict[str, object]]:
    def guardrail_node(state: GraphState) -> dict[str, object]:
        flags: list[str] = list(state.get("guardrail_flags", []))
        draft = state.get("draft_answer")
        iteration_count = state.get("iteration_count", 0)
        tool_agent_pending = state.get("intent") is not None and not state.get("tool_agent_done", True)

        if iteration_count >= settings.max_agent_iterations and tool_agent_pending:
            final_answer = ITERATION_LIMIT_MESSAGE
            flags.append("ESCALATED_ITERATION_LIMIT")
        elif not draft:
            final_answer = NO_DRAFT_FALLBACK_MESSAGE
            flags.append("NO_DRAFT_PRODUCED")
        elif _contains_advice_language(draft):
            final_answer = FINANCIAL_ADVICE_DISCLAIMER
            flags.append("FINANCIAL_ADVICE_BLOCKED")
        else:
            final_answer, was_redacted = _redact_sensitive_numbers(draft)
            if was_redacted:
                flags.append("PII_REDACTED")

        if flags:
            logger.info("guardrail_flags_raised", flags=flags, conversation_id=state.get("conversation_id"))

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
