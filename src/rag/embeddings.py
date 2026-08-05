"""Embedding backends for the retrieval pipeline.

`FakeHashEmbeddings` is a deterministic, offline, zero-dependency stand-in for
a real embedding model. It exists so the whole system (ingestion, retrieval,
the RAG agent, CI) can run end-to-end with zero API keys and zero network
access — a hard requirement for a portfolio project that must be cloned and
demoed by someone without DemoBank credentials. Its retrieval quality is
obviously worse than a real embedding model (it has no notion of synonymy or
semantics, only shared tokens): that trade-off is deliberate and documented
here, not hidden. `EMBEDDING_PROVIDER=openai` swaps in real embeddings without
touching any calling code.
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from app.core.config import EmbeddingProvider, Settings

_DIMENSIONS = 384


class FakeHashEmbeddings(Embeddings):
    """Feature-hashing bag-of-words embedding, L2-normalized for cosine similarity.

    Each token is hashed into one of `_DIMENSIONS` buckets with a deterministic
    sign, so two texts sharing tokens end up with positive cosine similarity
    and unrelated texts land near-orthogonal — enough signal to exercise the
    hybrid retriever and reranker without a real model.
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
    """Real embeddings when configured and a key is present, fake otherwise.

    Mirrors `app.core.llm.get_chat_model`'s fail-open-to-fake behaviour: a
    misconfigured or key-less environment should still boot and serve
    (degraded) traffic rather than crash at import time.
    """
    if settings.embedding_provider == EmbeddingProvider.OPENAI and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        from pydantic import SecretStr

        assert settings.openai_api_key is not None  # narrows for mypy; already checked above
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=SecretStr(settings.openai_api_key),
        )

    return FakeHashEmbeddings()
