"""Handles IntentLabel.SMALL_TALK — greetings, thanks, chit-chat.

Kept as its own node (rather than folded into the RAG agent) so it never has
retrieval or tools in its context: a smalltalk turn has no business citing a
policy document or calling `get_balance`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.prompts.smalltalk_prompt import SMALLTALK_SYSTEM_PROMPT
from agents.state import GraphState
from schemas.dto import AgentTraceStep


def build_smalltalk_node(llm: BaseChatModel) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def smalltalk_node(state: GraphState) -> dict[str, object]:
        response = await llm.ainvoke(
            [
                SystemMessage(content=SMALLTALK_SYSTEM_PROMPT),
                HumanMessage(content=state["user_query"]),
            ]
        )
        return {
            "draft_answer": str(response.content),
            "trace": [AgentTraceStep(node="smalltalk", summary="generated conversational reply")],
        }

    return smalltalk_node
