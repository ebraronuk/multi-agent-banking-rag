"""Lexical (BM25) reranking blended with vector similarity.

Vector similarity alone misses exact-term matches — a customer asking about
"FAST" or "KVKK" benefits from a lexical signal that rewards literal token
overlap, which embeddings (especially the fake hash embedding used offline)
can under-weight. Blending 50/50 rather than picking one covers both semantic
and lexical matches without the cost/latency of a cross-encoder, which this
demo's scale doesn't justify.
"""

from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from schemas.dto import Citation

_SNIPPET_LENGTH = 200


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lowest, highest = min(values), max(values)
    if highest == lowest:
        return [1.0 for _ in values]
    return [(value - lowest) / (highest - lowest) for value in values]


def _to_citation(document: Document, score: float) -> Citation:
    metadata = document.metadata
    return Citation(
        doc_id=str(metadata.get("doc_id", "")),
        title=str(metadata.get("title", "")),
        source=str(metadata.get("source", "")),
        snippet=document.page_content[:_SNIPPET_LENGTH],
        score=max(0.0, min(1.0, score)),
    )


def rerank_with_bm25(
    query: str,
    candidates: list[tuple[Document, float]],
    top_k: int = 4,
) -> list[Citation]:
    if not candidates:
        return []

    # BM25 needs at least two documents to produce a meaningful corpus
    # statistic (IDF is undefined/degenerate over a single document); with
    # 0-1 candidates there is nothing to rerank against, so fall back to the
    # vector score as-is.
    if len(candidates) == 1:
        document, vector_score = candidates[0]
        return [_to_citation(document, vector_score)]

    documents = [document for document, _ in candidates]
    vector_scores = [score for _, score in candidates]

    tokenized_corpus = [document.page_content.lower().split() for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = list(bm25.get_scores(query.lower().split()))

    normalized_vector = _min_max_normalize(vector_scores)
    normalized_bm25 = _min_max_normalize(bm25_scores)
    combined_scores = [
        0.5 * vector + 0.5 * bm25_score
        for vector, bm25_score in zip(normalized_vector, normalized_bm25, strict=True)
    ]

    ranked = sorted(
        zip(documents, combined_scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [_to_citation(document, score) for document, score in ranked[:top_k]]
