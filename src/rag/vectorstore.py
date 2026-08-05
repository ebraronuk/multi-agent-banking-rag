"""Chroma vector store factory.

A single function rather than a wrapping class: `Chroma` already carries all
the state a caller needs (persistence dir, collection, embedding function),
so a class here would only add indirection over the library's own API.
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
        # Chroma defaults new collections to squared-L2 distance, whose
        # "relevance score" isn't bounded to [0, 1] (negative values are
        # possible) — cosine space is what `similarity_search_with_relevance_scores`
        # assumes, and matches the L2-normalized vectors `FakeHashEmbeddings`
        # produces.
        collection_metadata={"hnsw:space": "cosine"},
    )
