"""Entegrasyon testleri: gerçek graf bağlantısı (agents/graph.py), sahte
LLM/embedding, ağ yok. Bunlar mock değil gerçek FastAPI uygulaması + derlenmiş
LangGraph grafiğini çalıştırıyor — amaç, tek tek düğüm testlerinin
göremeyeceği bağlantı hatalarını (yanlış düğüm adı, eksik kenar) yakalamak.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from mcp_server.tools.banking_repository import _fallback_repository


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
    # Test ortamında vektör deposu seed'lenmemiş olabilir (scripts/seed_vectorstore.py
    # ayrı, açık bir adım) — alıntılar boş olabilir, ama trace rag_agent'ın
    # gerçekten çalıştığını göstermeli, sessizce çökmediğini değil.
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
    # FastAPI'nin varsayılan doğrulama-hatası şekli değil — app/main.py'deki
    # validation_exception_handler tarafından kendi ErrorResponse
    # sözleşmemize dönüştürülmüş, diğer her hata yolunun döndürdüğü şeyin aynısı.
    assert body["code"] == "VALIDATION_ERROR"
    assert body["details"]["errors"]


async def test_chat_out_of_scope_gets_handoff_message(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": "yarın hava nasıl olacak?"})

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "OUT_OF_SCOPE"
    assert "yardımcı olabilirim" in body["answer"].lower()


async def test_every_response_carries_a_request_id_header(client: AsyncClient) -> None:
    response = await client.get("/healthz")

    assert "x-request-id" in response.headers
    # Çağıranın gönderdiği id yankılanmalı, değiştirilmemeli — bu, her hop'ta
    # kendi id'sini ekleyen servisler arasında isteğin izlenebilmesini sağlar.
    traced = await client.get("/healthz", headers={"X-Request-Id": "trace-abc-123"})
    assert traced.headers["x-request-id"] == "trace-abc-123"


async def test_chat_rate_limit_returns_429_after_the_configured_burst(
    client: AsyncClient,
) -> None:
    # Route @limiter.limit(settings.chat_rate_limit) ile dekore edilmiş,
    # varsayılan "20/minute" (bkz. app/core/config.py); sıkı bir döngüde
    # bir fazlasını ateşlemek tetiklemeli — /chat'in arkasındaki her LLM
    # çağrısı gerçek para demek, bu formalite değil gerçek bir koruma.
    responses = [await client.post("/chat", json={"message": "merhaba"}) for _ in range(21)]

    assert responses[-1].status_code == 429
    body = responses[-1].json()
    assert body["code"] == "RATE_LIMITED"
    assert all(r.status_code == 200 for r in responses[:20])


async def test_metrics_endpoint_exposes_prometheus_format(client: AsyncClient) -> None:
    await client.get("/healthz")  # instrumente edilmiş en az bir istek olduğundan emin ol

    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text


async def test_multi_turn_slot_fill_completes_a_card_action_across_two_requests(
    client: AsyncClient,
) -> None:
    """ADR-008'in var olma sebebi olan senaryo: aynı konuşmadaki çıplak bir
    takip cevabı, alakasız yeni bir mesaj gibi değil, isteği gerçekten
    tamamlayan bir cevap gibi işlenmeli."""
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

    # "4321" banking_repository.py'deki gerçek fixture kartı — rastgele
    # görünen bir sayı değil, bilinçli seçildi. Çıplak rakamların
    # ner_extractor'ın normalde tutunacağı bir anahtar kelimesi yok — bu
    # sadece memory_load ilk turdan bekleyen isteği taşıdığı için çalışıyor
    # (bkz. ADR-008).
    assert second_body["intent"] == "CARD_ACTION"
    assert len(second_body["tool_calls"]) == 1
    assert second_body["tool_calls"][0]["tool_name"] == "block_card"
    assert second_body["tool_calls"][0]["ok"] is True

    node_sequence = [step["node"] for step in second_body["trace"]]
    assert "memory_load" in node_sequence
    assert "tool_agent" in node_sequence

    # fixture durumunu geri al ki bu test başkalarına sızmasın
    for account in _fallback_repository._accounts.values():
        for card in account["cards"]:  # type: ignore[union-attr]
            if card["last4"] == "4321":
                card["status"] = "active"
