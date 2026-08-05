# ADR-005: Araç çağırma sınırı olarak MCP (FastMCP)

## Bağlam
`tool_agent` düğümünün bakiye sorgulama, işlem geçmişi, kart bloklama gibi "gerçek
dünyaya etki eden" işlemleri çağırması gerekiyor. Bu işlemler bugün mock/fixture veri
üzerinden çalışıyor ama sınırın nerede çizildiği ileride gerçek bir core-banking
entegrasyonuna geçişi kolaylaştırıp zorlaştırmayacağını belirliyor.

## Seçenekler
- **A: Doğrudan Python fonksiyon çağrısı** (`tool_agent` doğrudan `banking_tools.get_balance(...)`
  import edip çağırır). En basit, ama araçlar ile ajan aynı process/aynı dil ile
  sıkı bağlı kalır; araçları ayrı bir ekip/servis olarak ölçeklemek zorlaşır.
- **B: FastMCP üzerinden MCP protokolü** — araçlar ayrı bir process (`mcp_server/server.py`)
  olarak çalışır, ajan onlara ağ üzerinden (MCP client) bağlanır. İş ilanının da açıkça
  aradığı bir teknoloji (FastMCP), ve gerçek dünyada "araç sağlayıcı takım" ile
  "ajan takımı" farklı ekipler olduğunda doğal sınırı bu şekilde çiziyor.

## Tercih
**B, ama in-process bir düşüş (fallback) yolu ile.** `agents/tools/mcp_client.py` iki
sınıf sunuyor: `MCPToolClient` (gerçek ağ üzerinden FastMCP) ve `InProcessToolClient`
(aynı fonksiyonları process içi çağıran, testlerde ve `LLM_PROVIDER=fake` modunda
kullanılan hızlı yol). `get_tool_client(settings)` hangisinin kullanılacağına karar
veriyor. Böylece hem MCP'nin gösterdiği "araçlar ayrı bir servis" mimarisi hem de
testlerin ağ/sunucu ayağa kaldırmadan çalışabilmesi bir arada sağlanıyor.

## Sonuçlar
- ✅ İş ilanının aradığı FastMCP entegrasyonu gerçek, çalışan bir örnekle gösteriliyor.
- ✅ Birim testler bir MCP sunucusu ayağa kaldırmadan (`InProcessToolClient`) çalışıyor.
- ❌ İki client sınıfını senkron tutma yükü var (aynı `ToolCallRecord` sözleşmesine uymaları
  gerekiyor) — kabul edilebilir, çünkü bu tam da "iki path'in de gerçekten çalıştığını"
  garanti eden şey.
