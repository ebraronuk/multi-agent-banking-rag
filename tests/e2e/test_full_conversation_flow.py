"""One true end-to-end test: seed the (fake-embedding) vector store, then ask
a policy question through the real HTTP app and confirm a citation actually
comes back. Everything else is covered at the unit/integration layer; this is
the "does the whole thing actually work together" smoke test the playbook asks
for (ENGINEERING-STANDARDS-WEB.md §5: at least one e2e happy path)."""

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
    assert len(body["citations"]) > 0, "seeded KB should produce at least one citation"
    assert body["answer"]
    assert body["guardrail_flags"] == [] or "PII_REDACTED" in body["guardrail_flags"]

    node_sequence = [step["node"] for step in body["trace"]]
    assert node_sequence == ["ner_agent", "intent_agent", "supervisor", "rag_agent", "guardrail"]
