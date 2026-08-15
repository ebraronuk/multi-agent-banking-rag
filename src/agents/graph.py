"""Her worker düğümünü tek bir derlenmiş LangGraph'a bağlar.

`build_graph(settings)`, sistemdeki her modülü (RAG, tools/MCP, NLP, guardrail)
tanıyan tek yer — işi bağımlılık enjeksiyonu + kenar kablolama, iş kuralı değil.
Diyagram için bkz. docs/architecture.md.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.memory import get_conversation_memory
from agents.state import GraphState
from agents.supervisor import (
    NODE_ADVANCE_INTENT,
    NODE_ESCALATE,
    NODE_GUARDRAIL,
    NODE_RAG_AGENT,
    NODE_SMALLTALK,
    NODE_SYNTHESIZER,
    NODE_TOOL_AGENT,
    advance_intent_node,
    build_supervisor_router,
    supervisor_node,
)
from agents.tools.mcp_client import get_tool_client
from agents.workers.escalate_agent import escalate_node
from agents.workers.guardrail_agent import build_guardrail_node
from agents.workers.intent_agent import build_intent_node
from agents.workers.memory_agent import build_memory_load_node, build_memory_save_node
from agents.workers.ner_agent import build_ner_node
from agents.workers.rag_agent import build_rag_node
from agents.workers.smalltalk_agent import build_smalltalk_node
from agents.workers.synthesizer_agent import build_synthesizer_node
from agents.workers.tool_agent import build_tool_agent_node
from app.core.config import Settings
from app.core.llm import get_chat_model
from rag.retriever import build_retriever

NODE_MEMORY_LOAD = "memory_load"
NODE_MEMORY_SAVE = "memory_save"
NODE_NER = "ner_agent"
NODE_INTENT = "intent_agent"
NODE_SUPERVISOR = "supervisor"


def build_graph(settings: Settings) -> CompiledStateGraph:
    llm = get_chat_model(settings)
    retriever = build_retriever(settings)
    tool_client = get_tool_client(settings)
    memory = get_conversation_memory(settings)

    graph = StateGraph(GraphState)

    # mypy, factory'den dönen `Callable[...]` tipli düğümler için GraphState'in
    # StateLike'ı sağladığını kanıtlayamıyor (gerçek bir tip hatası değil) —
    # her factory-built düğüm bu yüzden ignore alıyor, bare-function'lar almıyor.
    graph.add_node(NODE_MEMORY_LOAD, build_memory_load_node(memory))  # type: ignore[call-overload]
    graph.add_node(NODE_NER, build_ner_node(llm))  # type: ignore[call-overload]
    graph.add_node(NODE_INTENT, build_intent_node(llm))  # type: ignore[call-overload]
    graph.add_node(NODE_SUPERVISOR, supervisor_node)
    graph.add_node(NODE_RAG_AGENT, build_rag_node(retriever, llm))  # type: ignore[arg-type]
    tool_agent_node = build_tool_agent_node(tool_client, llm, settings)
    graph.add_node(NODE_TOOL_AGENT, tool_agent_node)  # type: ignore[arg-type]
    graph.add_node(NODE_SMALLTALK, build_smalltalk_node(llm))  # type: ignore[arg-type]
    graph.add_node(NODE_ESCALATE, escalate_node)
    graph.add_node(NODE_ADVANCE_INTENT, advance_intent_node)
    graph.add_node(NODE_SYNTHESIZER, build_synthesizer_node(llm))  # type: ignore[arg-type]
    graph.add_node(NODE_GUARDRAIL, build_guardrail_node(settings))  # type: ignore[arg-type]
    graph.add_node(NODE_MEMORY_SAVE, build_memory_save_node(memory, settings))  # type: ignore[arg-type]

    graph.add_edge(START, NODE_MEMORY_LOAD)
    graph.add_edge(NODE_MEMORY_LOAD, NODE_NER)
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
            NODE_ADVANCE_INTENT: NODE_ADVANCE_INTENT,
            NODE_SYNTHESIZER: NODE_SYNTHESIZER,
            NODE_GUARDRAIL: NODE_GUARDRAIL,
        },
    )

    # rag/tool_agent/smalltalk supervisor'a geri döner (extra_intents kuyruğu
    # kontrolü için, ADR-012). escalate zincire dahil değil, direkt guardrail'e düşer.
    graph.add_edge(NODE_RAG_AGENT, NODE_SUPERVISOR)
    graph.add_edge(NODE_TOOL_AGENT, NODE_SUPERVISOR)
    graph.add_edge(NODE_SMALLTALK, NODE_SUPERVISOR)
    graph.add_edge(NODE_ESCALATE, NODE_GUARDRAIL)
    graph.add_edge(NODE_ADVANCE_INTENT, NODE_SUPERVISOR)
    graph.add_edge(NODE_SYNTHESIZER, NODE_GUARDRAIL)
    graph.add_edge(NODE_GUARDRAIL, NODE_MEMORY_SAVE)
    graph.add_edge(NODE_MEMORY_SAVE, END)

    return graph.compile()
