"""`agents/workers/rag_agent.py` için birim testler."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.state import new_state
from agents.workers.rag_agent import _build_context_block, build_rag_node
from app.core.llm import FakeChatModel
from schemas.dto import Citation


class _RecordingRetriever:
    def __init__(self, citations: list[Citation] | None = None) -> None:
        self.queries: list[str] = []
        self._citations = citations

    def retrieve(self, query: str) -> list[Citation]:
        self.queries.append(query)
        if self._citations is not None:
            return self._citations
        return [Citation(doc_id="d1", title="Test", source="test.md", snippet="...", score=0.9)]


class _RecordingLLM:
    """Gönderilen mesajları yakalayan stub — asıl bir cevap üretmesi gerekmiyor."""

    def __init__(self) -> None:
        self.last_messages: list[object] = []

    async def ainvoke(self, messages: object) -> AIMessage:
        self.last_messages = list(messages)  # type: ignore[arg-type]
        return AIMessage(content="cevap")


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


def test_build_context_block_signals_no_context_explicitly_on_empty_retrieval() -> None:
    # 2026 RAG hatalarının çoğunluğu sessiz retrieval başarısızlığı — model,
    # boş bir "Bağlam:\n\n" bölümünden "cevap yok" çıkarımını her zaman
    # yapmıyor. Açık bir işaret, modelin kendi bilgisiyle doldurmasını
    # (halüsinasyon) zorlaştırır.
    assert _build_context_block([]) == "(İlgili bağlam bulunamadı.)"


def test_build_context_block_lists_citations_when_present() -> None:
    citations = [Citation(doc_id="d1", title="Test", source="test.md", snippet="içerik", score=0.9)]
    assert _build_context_block(citations) == "[1] Test: içerik"


async def test_rag_node_prompt_carries_explicit_no_context_marker_on_empty_retrieval() -> None:
    retriever = _RecordingRetriever(citations=[])
    llm = _RecordingLLM()
    node = build_rag_node(retriever, llm)  # type: ignore[arg-type]
    state = new_state("c1", "Alakasız bir soru?")

    await node(state)

    human_message = llm.last_messages[-1]
    assert "(İlgili bağlam bulunamadı.)" in human_message.content  # type: ignore[union-attr]
