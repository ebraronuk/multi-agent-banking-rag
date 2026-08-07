"""Tek mesajda birden fazla niyet işlendiğinde toplanan taslakları tek cevaba birleştirir (ADR-012).

Sadece `extra_intents` dolu geldiğinde (yani bileşik bir istek olduğunda) çalışan bir yol —
tek-niyetli turlarda supervisor bu düğüme hiç uğramıyor, ek maliyet yok. Fake modda taslaklar
art arda ekleniyor (deterministik, testte kolay doğrulanır); gerçek modda `tool_agent`/`rag_agent`
gibi `is_fake_model` build-time'da kontrol edilip gerçek bir birleştirme çağrısı yapılıyor.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts.synthesizer_prompt import SYNTHESIZER_SYSTEM_PROMPT
from agents.state import GraphState
from app.core.llm import is_fake_model, safe_ainvoke
from schemas.dto import AgentTraceStep


def _concatenate(drafts: list[str]) -> str:
    return "\n\n".join(drafts)


def build_synthesizer_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    use_llm_merge = not is_fake_model(llm)

    async def synthesizer_node(state: GraphState) -> dict[str, object]:
        drafts = list(state.get("collected_drafts", []))
        last_draft = state.get("draft_answer")
        if last_draft:
            drafts.append(last_draft)

        final_answer = None
        if use_llm_merge:
            final_answer = await safe_ainvoke(
                llm,
                [
                    SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
                    HumanMessage(content="\n\n---\n\n".join(drafts)),
                ],
                node="synthesizer",
            )
        final_answer = final_answer or _concatenate(drafts)

        return {
            "draft_answer": final_answer,
            "trace": [
                AgentTraceStep(node="synthesizer", summary=f"merged {len(drafts)} draft answer(s)")
            ],
        }

    return synthesizer_node
