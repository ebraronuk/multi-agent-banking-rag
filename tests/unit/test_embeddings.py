from __future__ import annotations

from app.core.config import EmbeddingProvider, Settings
from rag.embeddings import FakeHashEmbeddings, get_embeddings


def test_google_provider_without_key_falls_back_to_fake() -> None:
    # Eksik anahtarla cokmek yerine fake'e dusuyor: LLM tarafindaki
    # resolved_llm_provider ile ayni fail-open davranisi.
    settings = Settings(embedding_provider=EmbeddingProvider.GOOGLE, google_api_key=None)

    assert isinstance(get_embeddings(settings), FakeHashEmbeddings)


def test_google_provider_with_key_returns_google_embeddings() -> None:
    settings = Settings(
        embedding_provider=EmbeddingProvider.GOOGLE, google_api_key="test-anahtari"
    )

    embeddings = get_embeddings(settings)

    assert not isinstance(embeddings, FakeHashEmbeddings)
    assert type(embeddings).__name__ == "GoogleGenerativeAIEmbeddings"
