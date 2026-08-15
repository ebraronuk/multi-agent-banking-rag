"""Chroma vektör deposu factory'si.

Sarmalayan bir sınıf yerine tek bir fonksiyon: `Chroma` zaten bir çağıranın
ihtiyaç duyduğu tüm durumu (kalıcılık dizini, koleksiyon, embedding
fonksiyonu) taşıyor, burada bir sınıf kütüphanenin kendi API'sinin üzerine
sadece bir dolaylama katmanı eklerdi.
"""

from __future__ import annotations

from langchain_chroma import Chroma

from app.core.config import Settings
from rag.embeddings import get_embeddings


def build_vectorstore(settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(settings),
        persist_directory=settings.chroma_persist_dir,
        # Varsayılan squared-L2 mesafesinin relevance score'u [0,1]'e sınırlı değil;
        # `similarity_search_with_relevance_scores` cosine uzayını varsayıyor.
        collection_metadata={"hnsw:space": "cosine"},
    )
