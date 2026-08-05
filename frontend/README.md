# frontend

`multi-agent-banking-rag` backend'i için minimal bir sohbet arayüzü — Swagger'ın yerini
tutan, sistemin ayırt edici özelliğini (her turn'de hangi ajanın çalıştığını gösteren
`trace`) görsel bir "Ajan İzleme" panelinde canlandıran bir demo istemcisi.

Next.js (App Router) + TypeScript + Tailwind CSS + [Phosphor Icons](https://phosphoricons.com).

## Çalıştırma

Backend'in ayakta olması gerekiyor (repo kökünden):

```bash
docker compose -f ../docker/docker-compose.yml up --build
```

Sonra:

```bash
cp .env.local.example .env.local
npm install
npm run dev
```

`http://localhost:3000`'de açılır. Backend'in `CORS_ALLOWED_ORIGINS`'ı varsayılan olarak
`http://localhost:3000`'i zaten kabul ediyor (bkz. `../.env.example`).

## Ne gösteriyor

- Sohbet penceresi (sol) — mesaj gönderin, yanıtı görün.
- Ajan İzleme paneli (sağ) — seçili mesajın `trace` alanını, her ajanın hangi sırayla
  çalıştığını ikonlarla gösterir (`src/lib/nodeMeta.ts` — `agents/graph.py`'deki düğüm
  adlarıyla birebir eşleşir).
- Alıntılar (RAG), araç çağrıları (MCP) ve guardrail flag'leri, ilgili mesajın altında.

## Kapsam dışı (bilinçli)

Bu, backend'in `ChatResponse` sözleşmesini elle mirror'layan (`src/lib/types.ts`) küçük
bir demo istemcisi — kimlik doğrulama, mesaj düzenleme/silme, çoklu konuşma geçmişi gibi
gerçek bir ürün özelliği içermiyor; amaç backend'in çalıştığını görsel olarak kanıtlamak.
