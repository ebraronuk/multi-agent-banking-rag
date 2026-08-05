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
from mcp_server.tools.banking_tools import _ACCOUNTS


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
    body = response.json()
    # Not FastAPI's default validation-error shape — reshaped by
    # app/main.py::validation_exception_handler into our own ErrorResponse
    # contract, the same one every other error path returns.
    assert body["code"] == "VALIDATION_ERROR"
    assert body["details"]["errors"]


async def test_chat_out_of_scope_gets_handoff_message(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": "yarın hava nasıl olacak?"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "OUT_OF_SCOPE"
    assert "kapsam" in body["answer"].lower()


async def test_every_response_carries_a_request_id_header(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert "x-request-id" in response.headers
    # A caller-supplied id should be echoed back, not replaced — it's what
    # lets a request be traced across services that each add their own hop.
    traced = await client.get("/healthz", headers={"X-Request-Id": "trace-abc-123"})
    assert traced.headers["x-request-id"] == "trace-abc-123"


async def test_chat_rate_limit_returns_429_after_the_configured_burst(
    client: AsyncClient,
) -> None:
    # The route is decorated with @limiter.limit(settings.chat_rate_limit),
    # default "20/minute" (see app/core/config.py); firing one more than that
    # in a tight loop must trip it — every LLM call behind /chat costs real
    # money, so this is a real safeguard, not a formality.
    responses = [await client.post("/chat", json={"message": "merhaba"}) for _ in range(21)]

    assert responses[-1].status_code == 429
    body = responses[-1].json()
    assert body["code"] == "RATE_LIMITED"
    assert all(r.status_code == 200 for r in responses[:20])


async def test_metrics_endpoint_exposes_prometheus_format(client: AsyncClient) -> None:
    await client.get("/healthz")  # make sure at least one request has been instrumented

    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text


async def test_multi_turn_slot_fill_completes_a_card_action_across_two_requests(
    client: AsyncClient,
) -> None:
    """The scenario ADR-008 exists for: a bare follow-up answer, in the same
    conversation, actually completes the request instead of being treated as
    a brand-new, unrelated message."""
    first = await client.post(
        "/chat", json={"conversation_id": "multi-turn-1", "message": "kartımı blokla"}
    )
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["intent"] == "CARD_ACTION"
    assert "tool_calls" not in first_body or first_body["tool_calls"] == []
    assert any(step["node"] == "memory_save" for step in first_body["trace"])

    second = await client.post(
        "/chat", json={"conversation_id": "multi-turn-1", "message": "4321"}
    )
    assert second.status_code == 200
    second_body = second.json()

    # "4321" is the real fixture card in mcp_server/tools/banking_tools.py —
    # picked deliberately (not an arbitrary-looking number) so a real match
    # in _ACCOUNTS is exercised, not just the entity-synthesis mechanics.
    # The bare digits alone have no keyword ner_extractor would normally
    # anchor a CARD_LAST4 match on — this only works because memory_load
    # carried the pending request from the first turn (see ADR-008).
    assert second_body["intent"] == "CARD_ACTION"
    assert len(second_body["tool_calls"]) == 1
    assert second_body["tool_calls"][0]["tool_name"] == "block_card"
    assert second_body["tool_calls"][0]["ok"] is True

    node_sequence = [step["node"] for step in second_body["trace"]]
    assert "memory_load" in node_sequence
    assert "tool_agent" in node_sequence

    # restore fixture state so this test doesn't leak into others
    for account in _ACCOUNTS.values():
        for card in account["cards"]:  # type: ignore[union-attr]
            if card["last4"] == "4321":
                card["status"] = "active"
