"""Hybrid (vector + BM25) retriever used by the RAG worker node.

Kept as a thin class over `Chroma` + `rerank_with_bm25` rather than folding
retrieval logic into the agent node itself, so it can be constructed once per
process (one Chroma connection, one embedding backend) and unit-tested
without spinning up LangGraph.
"""

from __future__ import annotations

from langchain_chroma import Chroma

from app.core.config import Settings
from app.core.logging import get_logger
from rag.reranker import rerank_with_bm25
from rag.vectorstore import build_vectorstore
from schemas.dto import Citation

logger = get_logger(__name__)


class HybridRetriever:
    def __init__(self, vectorstore: Chroma, k_vector: int = 8, k_final: int = 4) -> None:
        self.vectorstore = vectorstore
        self.k_vector = k_vector
        self.k_final = k_final

    def retrieve(self, query: str) -> list[Citation]:
        try:
            candidates = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=self.k_vector
            )
        except Exception:
            logger.exception("vector_search_failed", query=query)
            return []

        return rerank_with_bm25(query, candidates, top_k=self.k_final)


def build_retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(build_vectorstore(settings))
