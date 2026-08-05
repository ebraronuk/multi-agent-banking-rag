"""Wires every worker node into a single compiled LangGraph.

`build_graph(settings)` is the one place that knows about every module in the
system — RAG, tools/MCP, NLP, guardrails. Everything it imports exposes a
`build_x_node(...)` factory that closes over its own dependencies, so this
file's job is dependency injection + edge wiring, never business logic. See
docs/architecture.md for the diagram this function literally builds.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.state import GraphState
from agents.supervisor import (
    NODE_ESCALATE,
    NODE_GUARDRAIL,
    NODE_RAG_AGENT,
    NODE_SMALLTALK,
    NODE_TOOL_AGENT,
    build_supervisor_router,
    supervisor_node,
)
from agents.tools.mcp_client import get_tool_client
from agents.workers.escalate_agent import escalate_node
from agents.workers.guardrail_agent import build_guardrail_node
from agents.workers.intent_agent import build_intent_node
from agents.workers.ner_agent import build_ner_node
from agents.workers.rag_agent import build_rag_node
from agents.workers.smalltalk_agent import build_smalltalk_node
from agents.workers.tool_agent import build_tool_agent_node
from app.core.config import Settings
from app.core.llm import get_chat_model
from rag.retriever import build_retriever

NODE_NER = "ner_agent"
NODE_INTENT = "intent_agent"
NODE_SUPERVISOR = "supervisor"


def build_graph(settings: Settings) -> CompiledStateGraph:
    llm = get_chat_model(settings)
    retriever = build_retriever(settings)
    tool_client = get_tool_client(settings)

    graph = StateGraph(GraphState)

    # mypy can't prove `GraphState` (a `TypedDict`) satisfies langgraph's
    # `StateLike` bound when the node comes back through a `Callable[...]`
    # -typed factory return value rather than a bare top-level `def` — a
    # structural-Protocol/TypedDict limitation in mypy itself, not a real type
    # error (the two bare-function nodes below need no ignore; every
    # factory-built one does). Runtime behaviour is verified by the full test
    # suite in tests/integration and tests/e2e, which exercises every node.
    graph.add_node(NODE_NER, build_ner_node())  # type: ignore[call-overload]
    graph.add_node(NODE_INTENT, build_intent_node(llm))  # type: ignore[call-overload]
    graph.add_node(NODE_SUPERVISOR, supervisor_node)
    graph.add_node(NODE_RAG_AGENT, build_rag_node(retriever, llm))  # type: ignore[arg-type]
    tool_agent_node = build_tool_agent_node(tool_client, llm, settings)
    graph.add_node(NODE_TOOL_AGENT, tool_agent_node)  # type: ignore[arg-type]
    graph.add_node(NODE_SMALLTALK, build_smalltalk_node(llm))  # type: ignore[arg-type]
    graph.add_node(NODE_ESCALATE, escalate_node)
    graph.add_node(NODE_GUARDRAIL, build_guardrail_node(settings))  # type: ignore[arg-type]

    graph.add_edge(START, NODE_NER)
    graph.add_edge(NODE_NER, NODE_INTENT)
    graph.add_edge(NODE_INTENT, NODE_SUPERVISOR)

    graph.add_conditional_edges(
        NODE_SUPERVISOR,
        build_supervisor_router(settings),
        {
            NODE_RAG_AGENT: NODE_RAG_AGENT,
            NODE_TOOL_AGENT: NODE_TOOL_AGENT,
            NODE_SMALLTALK: NODE_SMALLTALK,
            NODE_ESCALATE: NODE_ESCALATE,
            NODE_GUARDRAIL: NODE_GUARDRAIL,
        },
    )

    # rag/smalltalk/escalate are single-shot: they always fall through to the
    # guardrail. tool_agent is the only node that loops back through the
    # supervisor, since it may need another pass (see agents/supervisor.py).
    graph.add_edge(NODE_RAG_AGENT, NODE_GUARDRAIL)
    graph.add_edge(NODE_TOOL_AGENT, NODE_SUPERVISOR)
    graph.add_edge(NODE_SMALLTALK, NODE_GUARDRAIL)
    graph.add_edge(NODE_ESCALATE, NODE_GUARDRAIL)
    graph.add_edge(NODE_GUARDRAIL, END)

    return graph.compile()
