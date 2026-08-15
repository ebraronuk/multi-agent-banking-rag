"""Retrieval pipeline'ın embedding backend'leri.

`FakeHashEmbeddings`, tüm sistemin sıfır API anahtarı ve sıfır ağ erişimiyle
uçtan uca çalışabilmesi için var — kalitesi gerçek bir modelden düşük (sadece
paylaşılan token'lar, anlam bilgisi yok). `EMBEDDING_PROVIDER=openai` hiçbir
çağıran kodu değiştirmeden gerçek embedding'lere geçiyor.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from app.core.config import EmbeddingProvider, Settings
from nlp.text_utils import turkish_lower

_DIMENSIONS = 384


class FakeHashEmbeddings(Embeddings):
    """Feature-hashing bag-of-words embedding, cosine benzerliği için L2-normalize.

    Ortak token paylaşan iki metin pozitif cosine benzerliğine, alakasız
    metinler neredeyse ortogonale düşer — hibrit retriever/reranker'ı çalıştırmaya yeter.
    """

    def __init__(self, dimensions: int = _DIMENSIONS) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = turkish_lower(text).split() or [""]

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
    """Yapılandırılmış ve bir anahtar varsa gerçek embedding, yoksa fake
    (aynı fail-open davranışı, bkz. `app.core.llm.get_chat_model`)."""
    if settings.embedding_provider == EmbeddingProvider.OPENAI and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        assert settings.openai_api_key is not None  # mypy için daraltma; yukarıda zaten kontrol edildi
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=SecretStr(settings.openai_api_key),
        )

    return FakeHashEmbeddings()
