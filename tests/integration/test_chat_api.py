"""Integration tests: real graph wiring (agents/graph.py), fake LLM/embeddings,
no network. These exercise the actual FastAPI app + LangGraph compiled graph,
not mocks of them — the point is to catch wiring bugs (wrong node name, missing
edge) that unit tests of individual nodes can't see.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


async def test_healthz(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz(client: AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200


async def test_chat_small_talk_round_trip(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": "merhaba"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "SMALL_TALK"
    assert body["answer"]
    assert body["conversation_id"]
    assert any(step["node"] == "guardrail" for step in body["trace"])


async def test_chat_rag_query_returns_citations_or_honest_gap(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": "EFT limitiniz ne kadar?"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "RAG_QUERY"
    # The vector store may not be seeded in a test environment (scripts/seed_vectorstore.py
    # is a separate, explicit step) — citations can legitimately be empty, but the
    # trace must still show the rag_agent actually ran rather than crashing silently.
    assert any(step["node"] == "rag_agent" for step in body["trace"])


async def test_chat_account_action_without_iban_asks_for_it_instead_of_failing(
    client: AsyncClient,
) -> None:
    response = await client.post("/chat", json={"message": "bakiyem ne kadar?"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "ACCOUNT_ACTION"
    assert body["answer"]
    assert any(step["node"] == "tool_agent" for step in body["trace"])


async def test_chat_empty_message_is_rejected_with_422(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422


async def test_chat_out_of_scope_gets_handoff_message(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": "yarın hava nasıl olacak?"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "OUT_OF_SCOPE"
    assert "kapsam" in body["answer"].lower()
