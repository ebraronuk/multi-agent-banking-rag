"""Retrieval pipeline'ın embedding backend'leri.

`FakeHashEmbeddings`, gerçek bir embedding modelinin deterministik, offline,
sıfır-bağımlılıklı bir yerine geçeni. Tüm sistemin (ingestion, retrieval, RAG
ajanı, CI) sıfır API anahtarı ve sıfır ağ erişimiyle uçtan uca çalışabilmesi
için var. Retrieval kalitesi gerçek bir embedding modelinden açıkça daha
düşük (eş anlamlılık ya da anlam bilgisi yok, sadece paylaşılan token'lar) —
bu bilinçli bir tercih, gizlenmiyor. `EMBEDDING_PROVIDER=openai`, hiçbir
çağıran kodu değiştirmeden gerçek embedding'lere geçiyor.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from app.core.config import EmbeddingProvider, Settings

_DIMENSIONS = 384


class FakeHashEmbeddings(Embeddings):
    """Feature-hashing bag-of-words embedding, cosine benzerliği için L2-normalize.

    Her token, deterministik bir işaretle `_DIMENSIONS` bucket'ından birine
    hash'leniyor — ortak token paylaşan iki metin pozitif cosine benzerliğine,
    alakasız metinler neredeyse ortogonale düşüyor. Gerçek bir model olmadan
    hibrit retriever'ı ve reranker'ı çalıştırmaya yetecek kadar sinyal.
    """

    def __init__(self, dimensions: int = _DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = text.lower().split() or [""]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], byteorder="big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]


def get_embeddings(settings: Settings) -> Embeddings:
    """Yapılandırılmış ve bir anahtar varsa gerçek embedding, yoksa fake.

    `app.core.llm.get_chat_model`'in fail-open-to-fake davranışını
    yansıtıyor: yanlış yapılandırılmış ya da anahtarsız bir ortam import
    zamanında çökmek yerine yine de (düşük kaliteli) trafiği servis etmeli.
    """
    if settings.embedding_provider == EmbeddingProvider.OPENAI and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        assert settings.openai_api_key is not None  # mypy için daraltma; yukarıda zaten kontrol edildi
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=SecretStr(settings.openai_api_key),
        )

    return FakeHashEmbeddings()
