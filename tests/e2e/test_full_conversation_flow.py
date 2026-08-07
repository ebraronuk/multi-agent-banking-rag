"""Tek gerçek uçtan-uca test: (fake-embedding) vektör deposunu seed'le, gerçek
HTTP uygulaması üzerinden bir politika sorusu sor ve gerçekten bir alıntı
döndüğünü doğrula. Geri kalan her şey birim/entegrasyon katmanında kapsanıyor;
bu "her şey gerçekten birlikte çalışıyor mu" duman testi — en az bir e2e
happy-path senaryosu."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import get_settings
from app.main import app
from rag.ingest import load_sample_documents
from rag.vectorstore import build_vectorstore


@pytest.fixture
async def seeded_client() -> AsyncIterator[AsyncClient]:
    settings = get_settings()
    vectorstore = build_vectorstore(settings)
    vectorstore.add_documents(load_sample_documents())

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


async def test_policy_question_returns_a_grounded_citation(seeded_client: AsyncClient) -> None:
    response = await seeded_client.post(
        "/chat", json={"message": "Kartımı ne zaman bloke edebilirim, politikanız nedir?"}
    )

    assert response.status_code == 200
    body = response.json()

    assert body["intent"] == "RAG_QUERY"
    assert len(body["citations"]) > 0, "seed'lenmiş bilgi tabanı en az bir alıntı üretmeli"
    assert body["answer"]
    assert body["guardrail_flags"] == [] or "PII_REDACTED" in body["guardrail_flags"]

    node_sequence = [step["node"] for step in body["trace"]]
    assert node_sequence == [
        "memory_load",
        "ner_agent",
        "intent_agent",
        "supervisor",
        "rag_agent",
        "supervisor",  # rag_agent bitince supervisor'a döner — kuyrukta ek niyet
        # yoksa (fake modda hiç olmuyor) doğrudan guardrail'e düşer (ADR-012).
        "guardrail",
        "memory_save",
    ]
