# ADR-009: Gerçek LLM'de `bind_tools` akıl yürütme döngüsü — ADR-002'nin genişletilmesi

## Bağlam
ADR-002, rota kararının (hangi ajan çalışsın) LLM'e değil deterministik bir Python
fonksiyonuna bırakılmasına karar vermişti — gerekçe hâlâ geçerli: test edilebilirlik,
gecikme/maliyet. Ama `tool_agent`'ın kendisi de (hangi *araç* çağrılsın) aynı şekilde
tamamen deterministik bir `_INTENT_TOOL_MAP`'e dayanıyordu; bu, tek-araçlı istekler için
doğru ama **bileşik** bir istekte ("kartımı blokla **ve** bir destek talebi aç") yetersiz
kalıyor — sabit harita turn başına tek bir (intent → araç) eşlemesi tanımlıyor, iki farklı
aracı sırayla çağırmayı hiç ifade edemiyor. İş ilanının da açıkça istediği şey tam olarak
bu: "LLM tabanlı ajanların planlama, akıl yürütme ve görev paylaşımı yetkinliği."

## Seçenekler
- **A: Deterministik haritayı büyütmek** — "hem X hem Y" gibi bileşik durumlar için elle
  yeni kombinasyonlar eklemek. Kombinasyon sayısı araç sayısıyla katlanarak büyür,
  gerçek bir planlama yeteneği göstermez, sadece daha büyük bir if/elif olur.
- **B: Her zaman LLM'e planlattırmak** (fake modda dahil) — ADR-002'nin argümanını
  tamamen terk etmek olurdu; `FakeChatModel`'in `bind_tools` çıktısı anlamlı değil
  (bir hash digest'in "hangi aracı çağırayım" diye akıl yürütmesi mümkün değil).
- **C: İki yol da var olsun** — `FakeChatModel` (offline/CI/anahtarsız) deterministik
  haritayı kullanmaya devam eder; gerçek bir Anthropic/OpenAI modeli bağlıyken
  `tool_agent`, `bind_tools` ile gerçek bir ReAct-tarzı döngüye geçer.

## Tercih
**C.** `build_tool_agent_node` artık `is_fake_model(llm)`'e göre dispatch ediyor
(`agents/workers/tool_agent.py`):
- **Deterministik** (`_deterministic_tool_call`) — değişmedi, ADR-002'nin orijinal
  gerekçesi hâlâ geçerli: fake modda ve tek-araçlı, kesinlik gerektiren yol için.
- **LLM-planlı** (`_reasoning_tool_call`) — `get_balance`/`list_transactions`/
  `block_card`/`open_support_ticket` gerçek `@tool`-dekore edilmiş fonksiyonlar olarak
  `bind_tools`'a veriliyor; model önerdiği her araç çağrısı gerçekten çalıştırılıp
  sonucu (`ToolMessage`) modele geri besleniyor, model "daha fazla araca gerek yok"
  deyip düz metin döndürene kadar bu devam ediyor — `settings.max_agent_iterations`
  ile sınırlı (bir model sonsuza kadar "bir araç daha" isteyemez).

**Kritik güvenlik katmanı — argüman doğrulama (`_validate_tool_args`):** Bir model,
`account_id`/`card_last4` gibi alanlara *herhangi bir* uydurma değer koyabilir — hiçbir
şey onu durdurmaz. Bu yüzden her önerilen araç çağrısı, gerçekten çalıştırılmadan önce
o turn'de `ner_agent`'ın (veya bir önceki turn'den taşınan slot-doldurmanın) bulduğu
gerçek varlıklarla karşılaştırılıyor (`_grounded_entity_values`); eşleşmiyorsa çağrı
reddediliyor ve model buna göre bilgilendiriliyor. "Model hangi aracı çağıracağına karar
versin" ile "model kimin hesabına dokunacağına karar versin" arasındaki çizgi burada —
ilki kabul edilebilir bir akıl yürütme, ikincisi asla değil (bkz.
`tests/unit/test_tool_agent.py::test_reasoning_loop_refuses_an_ungrounded_argument`).

## Sonuçlar
- ✅ Bileşik istekler ("kartımı blokla ve bir talep aç") tek turn'de, gerçek bir
  planlama/akıl yürütme döngüsüyle çözülüyor
  (`test_reasoning_loop_handles_a_compound_multi_tool_request_in_one_hop`).
- ✅ Argüman doğrulama, halüsinasyon riskini "hangi araç" seviyesinde tutuyor, "kimin
  parası/kartı" seviyesine hiç taşımıyor.
- ✅ Maksimum hop sınırı gerçekten test edildi (`test_reasoning_loop_stops_at_max_hops...`)
  — sonsuz döngü riski olmadan.
- ❌ Bu yol yalnızca gerçek bir API anahtarıyla anlamlı şekilde çalışır; `LLM_PROVIDER=fake`
  (varsayılan demo modu) hâlâ deterministik yolu kullanıyor. CI/testlerde bu yol
  `_ScriptedToolCallingModel` (gerçek bir sağlayıcı olmayan, senaryosu yazılmış bir test
  modeli) ile doğrulanıyor — döngünün *mekaniği* test ediliyor, bir modelin akıl
  yürütme *kalitesi* değil (o, Anthropic/OpenAI'ın sorumluluğu).
- ❌ İki ayrı kod yolunun (deterministik + LLM-planlı) senkron tutulması gerekiyor — yeni
  bir araç eklenirse hem `_INTENT_TOOL_MAP`'e hem `_build_tool_specs`'e eklenmeli. Kabul
  edilebilir bir bakım maliyeti: her iki yolun da gerçekten çalıştığını göstermenin bedeli.
