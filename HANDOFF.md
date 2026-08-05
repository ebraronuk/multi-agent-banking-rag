# HANDOFF

Bu depoya yeni katılan biri için: nereden başla, neye dikkat et.

## Nereden başla

1. [`README.md`](README.md) → hızlı başlangıç, `docker compose up`.
2. [`docs/architecture.md`](docs/architecture.md) → graph diyagramı + süreç sınırları.
3. [`docs/decisions/`](docs/decisions/) → "neden böyle" sorularının çoğunun cevabı burada.
4. `src/agents/graph.py` → gerçek wiring; hangi düğüm hangi düğüme bağlı, kodda tek bakışta.
5. `src/agents/state.py` → paylaşılan sözleşme; yeni bir alan eklerken önce burayı güncelle.

## Yeni bir intent/ajan eklerken

1. `schemas/dto.py::IntentLabel`'a yeni değeri ekle.
2. `nlp/intent_classifier.py::_INTENT_KEYWORDS`'e anahtar kelimeleri ekle (+ `intent_prompt.py`
   few-shot'una bir örnek).
3. `agents/supervisor.py::build_supervisor_router`'a yeni bir dal ekle.
4. Yeni ajanı `agents/workers/` altına, mevcut worker'lardaki imza kalıbını izleyerek yaz
   (`build_x_node(deps...) -> Callable[[GraphState], dict | Awaitable[dict]]`).
5. `agents/graph.py`'de düğümü ve kenarlarını ekle.
6. `tests/unit/` altına en az bir test; `tests/integration/test_chat_api.py`'ye bu intent'i
   tetikleyen bir senaryo ekle.

## Bilinen kırılganlıklar / dikkat edilecekler

- `FakeChatModel` (`app/core/llm.py`) yalnızca metin üretir; `tool_agent` bunun `bind_tools`
  ile anlamlı bir tool-call üretemeyeceğini bildiği için `is_fake_model` kontrolüyle
  deterministik bir entity→tool eşlemesine düşüyor. Yeni bir worker LLM'den yapılandırılmış
  çıktı bekliyorsa aynı deseni (fake path + gerçek path + `try/except` fallback) izlemeli.
- `GraphState`'teki liste alanları iki türlü davranıyor: `trace`/`tool_calls`
  `operator.add` reducer'ıyla **birikiyor** (bir düğüm sadece kendi yeni girdilerini
  döndürür), `entities`/`guardrail_flags` düz üzerine yazılıyor (sahibi tek düğüm).
  Karıştırmak sessiz veri kaybına yol açar — bir düğüm trace'i "tamamını" döndürürse
  önceki adımların izleri kaybolmaz (reducer zaten append eder) ama entities'i "eklemeye"
  çalışırsa (append reducer yokken) önceki turn'ün varlıkları sessizce üzerine yazılır.
- `scripts/seed_vectorstore.py` çalıştırılmadan RAG ajanı boş bir bilgi tabanına karşı
  çalışır — `citations=[]` döner, LLM'e "context'te yok" demesi söylenmiştir ama bu bir
  hata değil, sessiz bir boş sonuçtur. CI/demo öncesi seed adımını unutmayın.
- `MCP_SERVER_HOST` iki farklı anlamda kullanılıyor: `api` servisi bunu *bağlanılacak*
  hostname olarak okur (Docker DNS ile `mcp`), `mcp` servisinin kendisi ise aynı adı
  *bind edilecek* adres olarak okur. `docker-compose.yml`'de `mcp` servisi için bunu
  `0.0.0.0` olarak override etmezseniz, sunucu sadece kendi container'ının loopback'ine
  bağlanır ve `api`'den erişilemez hale gelir — sessizce "connection refused" verir,
  400/500 değil. Bu, Docker'da canlı çalıştırılıp gerçek ağ üzerinden test edilene kadar
  fark edilmemiş gerçek bir bug'dı (testler `InProcessToolClient` kullandığı için
  yakalayamadı) — bkz. `docker/docker-compose.yml` yorumu.
- FastMCP'nin `mcp.run()`'ı varsayılan olarak **stdio** transport'a düşer; `host`/`port`
  vermek için `transport="http"` açıkça geçilmeli (`mcp_server/server.py`), aksi halde
  `TransportMixin.run_stdio_async() got an unexpected keyword argument 'host'` hatasıyla
  container hemen çöker. HTTP transport'un varsayılan path'i de `/mcp` — `mcp_server_url`
  bunu içermezse client bağlanamaz.
- `data/vectorstore` bir Docker named volume'e mount ediliyor; imaj non-root `appuser`
  olarak çalıştığı için, Dockerfile bu dizini build sırasında oluşturup chown etmezse
  volume root-owned gelir ve Chroma "unable to open database file" ile başlangıçta çöker.

## Yapılmadı / bilinçli olarak ertelendi

`README.md`'nin "Sınırlar / sonraki adımlar" bölümüne bakın — kalıcı konuşma geçmişi,
çok adımlı tool planlama, auth, LangSmith tracing.
