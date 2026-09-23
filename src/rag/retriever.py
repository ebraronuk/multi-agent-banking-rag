"""RAG worker düğümünün kullandığı hibrit (vektör + BM25) retriever.

Retrieval mantığını ajan düğümüne gömmek yerine `Chroma` + `rerank_with_bm25`
üzerine ince bir sınıf olarak tutuluyor — süreç başına bir kere kurulabilsin
(tek bir Chroma bağlantısı, tek bir embedding backend'i) ve LangGraph
kurmadan birim test edilebilsin diye.
"""

from __future__ import annotations

import warnings

from langchain_chroma import Chroma

from app.core.config import Settings
from app.core.logging import get_logger
from rag.reranker import rerank_with_bm25
from rag.vectorstore import build_vectorstore
from schemas.dto import Citation

logger = get_logger(__name__)


class HybridRetriever:
    """Vektör aday havuzu + BM25 yeniden sıralama.

    `k_vector` bir performans ayarı gibi görünüyor ama değil: BM25 yalnızca
    bu havuzun içini sıralayabiliyor, dolayısıyla havuz aynı zamanda sözcüksel
    kanalın görebileceği en geniş küme. Dar bir havuz, hibrit mimariyi sessizce
    "zayıf embedding ne derse o" haline getiriyor — doğru doküman havuza
    girmediğinde reranker'ın onu kurtarma şansı yok (bkz. ADR-015).
    """

    def __init__(
        self,
        vectorstore: Chroma,
        k_vector: int = 24,
        k_final: int = 4,
        vector_weight: float = 0.25,
    ) -> None:
        self.vectorstore = vectorstore
        self.k_vector = k_vector
        self.k_final = k_final
        self.vector_weight = vector_weight

    def retrieve(self, query: str) -> list[Citation]:
        try:
            # Chroma, relevance skorunun [0,1] dışına çıktığı her koşuda
            # UserWarning veriyor. Burada bu beklenen bir durum: cosine
            # uzayında relevance = 1 - mesafe, yani [-1,1] aralığında ve
            # işaretli hash embedding'lerde negatif değerler gerçekten
            # oluşuyor (iki metin ortak token paylaşmayıp ters işaretli
            # kovalara düştüğünde). Skorlar zaten min-max normalize edildiği
            # için mutlak değer sıralamayı etkilemiyor.
            #
            # Negatif skorlu adayları elemek daha temiz görünürdü ama aynı
            # hatayı tekrar yapardı: sözcüksel kanalın göremediği bir doküman,
            # zayıf embedding yüzünden kaybolur.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Relevance scores must be between", category=UserWarning
                )
                candidates = self.vectorstore.similarity_search_with_relevance_scores(
                    query, k=self.k_vector
                )
        except Exception:
            logger.exception("vector_search_failed", query=query)
            return []

        return rerank_with_bm25(
            query, candidates, top_k=self.k_final, vector_weight=self.vector_weight
        )


def build_retriever(settings: Settings) -> HybridRetriever:
    return HybridRetriever(
        build_vectorstore(settings),
        k_vector=settings.rag_candidate_pool,
        vector_weight=settings.rag_vector_weight,
    )
