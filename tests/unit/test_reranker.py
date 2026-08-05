"""Unit tests for the BM25 + vector-score hybrid reranker.

No network, no vector store — candidates are hand-built (Document, score)
pairs, which is exactly the shape `HybridRetriever.retrieve` passes in after
`similarity_search_with_relevance_scores`.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.reranker import rerank_with_bm25


def _doc(doc_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"doc_id": doc_id, "title": doc_id, "source": f"{doc_id}.md"},
    )


def test_rerank_empty_candidates_returns_empty_list() -> None:
    assert rerank_with_bm25("kart engelleme", [], top_k=4) == []


def test_rerank_single_candidate_skips_bm25_and_uses_vector_score() -> None:
    document = _doc("only", "kart engelleme politikası hakkında bilgi")
    citations = rerank_with_bm25("kart engelleme", [(document, 0.73)], top_k=4)

    assert len(citations) == 1
    assert citations[0].doc_id == "only"
    assert citations[0].score == 0.73


def test_rerank_prefers_lexically_and_semantically_relevant_document() -> None:
    query = "kart engelleme nasıl yapılır"

    # "target" is the only candidate whose tokens overlap the query at all, so
    # it is the unique BM25-scoring document in this corpus (the other three
    # share zero query terms and therefore tie at a BM25 score of 0). Vector
    # scores are set so target does NOT have the best raw vector score either
    # (decoy does) — only the BM25 contribution can make target win, which is
    # exactly what this test is meant to prove about the 50/50 blend.
    target = _doc(
        "target", "kart engelleme kart engelleme mobil uygulama üzerinden anında yapılabilir"
    )
    decoy = _doc("decoy", "çalışma saatleri hafta içi dokuz on yedi")
    filler_1 = _doc("filler-1", "havale eft limitleri günlük tutar")
    filler_2 = _doc("filler-2", "hesap işletim ücreti aylık tutar")

    candidates = [
        (target, 0.45),
        (decoy, 0.90),
        (filler_1, 0.60),
        (filler_2, 0.30),
    ]

    citations = rerank_with_bm25(query, candidates, top_k=4)

    assert [citation.doc_id for citation in citations][0] == "target"
    assert all(0.0 <= citation.score <= 1.0 for citation in citations)


def test_rerank_respects_top_k() -> None:
    query = "hesap ücreti"
    candidates = [
        (_doc(f"doc-{i}", f"hesap işletim ücreti bilgisi numara {i}"), 0.1 * i)
        for i in range(1, 6)
    ]

    citations = rerank_with_bm25(query, candidates, top_k=2)

    assert len(citations) == 2


def test_rerank_snippet_is_truncated_to_200_chars() -> None:
    long_text = "kart engelleme " * 30
    document = _doc("long", long_text)

    citations = rerank_with_bm25("kart engelleme", [(document, 0.9)], top_k=4)

    assert len(citations[0].snippet) <= 200
