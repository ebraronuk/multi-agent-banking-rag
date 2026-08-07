"""IntentLabel.RAG_QUERY'i işler — bilgi tabanından yanıtlanabilecek politika/SSS soruları.

Retrieval ve üretim iki ayrı grafik adımına değil tek bir düğümde tutuluyor —
her zaman sıralı çalışıyorlar ve burada ayrı ayrı yeniden denenmiyorlar;
ayırmak dallanma açısından bir kazanç sağlamadan sadece bir state-aktarım
noktası eklerdi.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.memory import history_to_messages
from agents.prompts.rag_prompt import RAG_SYSTEM_PROMPT
from agents.state import GraphState
from app.core.llm import safe_ainvoke
from rag.retriever import HybridRetriever
from schemas.dto import AgentTraceStep, Citation


def _build_context_block(citations: list[Citation]) -> str:
    return "\n".join(
        f"[{index}] {citation.title}: {citation.snippet}"
        for index, citation in enumerate(citations, start=1)
    )


def build_rag_node(
    retriever: HybridRetriever, llm: BaseChatModel
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def rag_node(state: GraphState) -> dict[str, object]:
        citations = retriever.retrieve(state["user_query"])
        context = _build_context_block(citations)

        draft_answer = await safe_ainvoke(
            llm,
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                *history_to_messages(state.get("history", [])),
                HumanMessage(content=f"Bağlam:\n{context}\n\nSoru: {state['user_query']}"),
            ],
            node="rag_agent",
        )

        return {
            "retrieved_docs": citations,
            "draft_answer": draft_answer,
            "worker_pass_done": True,
            "trace": [
                AgentTraceStep(
                    node="rag_agent", summary=f"answered using {len(citations)} citation(s)"
                )
            ],
        }

    return rag_node
