"""IntentLabel.SMALL_TALK'ı işler — selamlama, teşekkür, sohbet.

RAG ajanına gömülmek yerine kendi düğümü olarak duruyor ki bağlamında hiçbir
zaman retrieval ya da araç olmasın: bir sohbet turn'ünün bir politika
dokümanına alıntı yapmaya ya da `get_balance` çağırmaya işi yok.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.memory import history_to_messages
from agents.prompts.smalltalk_prompt import SMALLTALK_SYSTEM_PROMPT
from agents.state import GraphState
from app.core.llm import safe_ainvoke
from schemas.dto import AgentTraceStep


def build_smalltalk_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def smalltalk_node(state: GraphState) -> dict[str, object]:
        draft_answer = await safe_ainvoke(
            llm,
            [
                SystemMessage(content=SMALLTALK_SYSTEM_PROMPT),
                *history_to_messages(state.get("history", [])),
                HumanMessage(content=state["user_query"]),
            ],
            node="smalltalk",
        )
        return {
            "draft_answer": draft_answer,
            "worker_pass_done": True,
            "trace": [AgentTraceStep(node="smalltalk", summary="generated conversational reply")],
        }

    return smalltalk_node
