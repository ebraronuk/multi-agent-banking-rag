"""`agents/workers/rag_agent.py` için birim testler."""

from __future__ import annotations

from agents.state import new_state
from agents.workers.rag_agent import build_rag_node
from app.core.llm import FakeChatModel
from schemas.dto import Citation


class _RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[Citation]:
        self.queries.append(query)
        return [Citation(doc_id="d1", title="Test", source="test.md", snippet="...", score=0.9)]


async def test_rag_node_uses_full_message_when_no_active_sub_query() -> None:
    retriever = _RecordingRetriever()
    node = build_rag_node(retriever, FakeChatModel())  # type: ignore[arg-type]
    state = new_state("c1", "EFT limitiniz ne kadar?")

    await node(state)

    assert retriever.queries == ["EFT limitiniz ne kadar?"]


async def test_rag_node_prefers_active_sub_query_over_full_message() -> None:
    # ADR-012: bileşik bir mesajda advance_intent_node bu alanı izole edilmiş
    # alt-sorguyla doldurmuşsa, rag_agent tam (gürültülü) mesaj yerine onu kullanmalı.
    retriever = _RecordingRetriever()
    node = build_rag_node(retriever, FakeChatModel())  # type: ignore[arg-type]
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["active_sub_query"] = "EFT limitiniz ne kadar"

    await node(state)

    assert retriever.queries == ["EFT limitiniz ne kadar"]
