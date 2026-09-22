# ADR-012: Tek mesajda birden fazla niyeti işleme (çoklu-niyet dispatch)

## Bağlam
`intent_agent` bir mesaja tek `IntentLabel` atıyor, `supervisor` tek worker'a yönlendiriyor.
"Kartımı blokla ve EFT limitiniz ne kadar?" gibi iki kategoriye giren bir istekte sistem
birini seçip diğerini hiç işlemiyor; kullanıcı ikinci kez yazmak zorunda kalıyor.

`tool_agent` zaten aynı kategori içinde birden fazla aracı bir turda çağırabiliyor
(ADR-009) — ama supervisor'ın seçtiği tek kategorinin içinde. RAG_QUERY ile CARD_ACTION
arasında köprü yok.

**Ölçülmüş kullanıcı verisi yok.** Bu bir portföy projesi; gerekçe tek-etiketli
sınıflandırmanın teorik sınırından geliyor, gözlemlenmiş bir talep oranından değil.

## Araştırma
- **Supervisor vs. swarm**: supervisor'da her etkileşim bir yönlendirme LLM çağrısından
  geçiyor; swarm'da ilk ajan karşılayamazsa doğrudan uzmana devrediyor.
- **Anthropic'in multi-agent araştırma sistemi**: lead agent planlıyor, alt-ajanları paralel
  çalıştırıyor, ayrı bir sentez adımıyla birleştiriyor. Kalite farkı ciddi ama token maliyeti
  ~15 kat — bu projeyle aynı ölçekte değil. Alınan tek şey: "birden fazla worker'ı işletip
  ayrı bir adımda birleştirme".

## Seçenekler
- **A: Tek niyet kalsın.** Senaryo çözülmüyor.
- **B: `Send` API'siyle paralel fan-out.** `iteration_count`/`tool_calls` gibi paylaşılan
  state alanlarının reducer'larla doğru birleşmesini garanti etmek, `tool_agent`'ın iç
  döngüsüyle birlikte gerçek bir karmaşıklık kaynağı.
- **C: Sıralı fan-out.** Supervisor'a "sıradaki niyet" kuyruğu; her worker bitince dönüyor,
  kuyruk boşalınca `synthesizer` taslakları birleştiriyor.

## Tercih
**C.** `tool_calls` zaten `operator.add` ile birikiyor, worker'larda hiçbir değişiklik
gerekmiyor, ve tek-niyetli mesajlarda (çoğunluk) ek LLM çağrısı yok — `intent_agent`'ın
mevcut structured-output çağrısına opsiyonel bir `extra_intents` alanı ekleniyor.

Kuyrukta niyet kalırsa `advance_intent_node` (LLM'siz) sıradakini aktif yapıyor, taslağı
`collected_drafts`'a itiyor. Kuyruk boşalınca birden fazla taslak varsa `synthesizer`
birleştiriyor, yoksa doğrudan guardrail'e gidiliyor — bugünkü davranışla birebir aynı.

`ESCALATE`/`OUT_OF_SCOPE` zincire dahil değil: insana aktarım isteği genelde konuşmanın
bittiği anlamına geliyor. `SMALL_TALK` ilk yazımda dışlanmıştı; entegrasyon testi ("EFT
limitiniz ne kadar, bu arada merhaba") kırmızı çıkınca dahil edildi — selamlama konuşmayı
bitirmiyor.

## Sonuçlar
- ✅ Bileşik istekler tek turda, tek cevapta karşılanıyor.
- ✅ Tek-niyetli mesajlarda davranış ve maliyet birebir aynı.
- ✅ İki kategoriden gelen araç çağrıları (`block_card` + `get_balance`) tek
  `ChatResponse.tool_calls`'ta doğru birikiyor.
- ✅ (sonradan) `extra_intents` sadece gerçek LLM'den geliyordu; `LLM_PROVIDER=fake`
  (anahtarsız `docker compose up` yolu) hep boş liste dönüyordu — turun en yeni parçası
  anahtarsız çalıştıranlarda hiç tetiklenmiyordu. `_rule_based_extra_intents` bunu kapattı.
- ✅ (sonradan) İlk kural tabanlı sürüm çok gevşekti: "Kartımı ne zaman bloke edebilirim,
  politikanız nedir?" saf bir RAG_QUERY'ydi ama sadece "bloke" geçtiği için CARD_ACTION da
  extra intent sayılıp cevaba alakasız bir "kartının son 4 hanesi?" sorusu ekleniyordu.
  Düzeltme: işlem tetikleyen niyetler artık bir entity ya da çok kelimeli spesifik bir
  kalıpla desteklenmeden extra intent sayılmıyor; RAG_QUERY/SMALL_TALK tek eşleşmeyle
  yetiniyor.
- ✅ (sonradan) Her alt-niyet `user_query`'nin tamamını görüyordu. Canlıda: bileşik mesajda
  `rag_agent`'ın retrieval sorgusu kart-blokaj kelimeleriyle kirlenip zayıf citation
  getiriyor, model boşluğu **yanlış rakam uydurarak** dolduruyordu — gerçek KB değeri
  50.000 TL iken "100.000 TL" dedi. `advance_intent_node` artık RAG_QUERY'ye geçerken
  mesajın o kısmını izole edip `active_sub_query`'ye yazıyor. Diğer niyetlere
  genişletilmedi; onlar entity-grounded çalıştığı için gürültüden aynı ölçüde
  etkilenmiyor.
- ❌ İkinci niyet, birincinin **sonucuna** bağlı olamıyor ("bakiyeme bak, düşükse uyarı
  ekle"). Her pass bağımsız. Çözümü ayrı bir ADR'yi hak eder.
- ❌ `max_agent_iterations` tüm turda tek sayaç — bileşik istek + uzun `tool_agent`
  döngüsü aynı anda olursa ikinci niyet limite takılabilir. Demo ölçeğinde gözlemlenmedi.
- ❌ Semantik (embedding tabanlı) router katmanı eklenmedi. Bilinçli: varsayılan
  `EMBEDDING_PROVIDER=fake` gerçek anlamsal benzerlik değil token örtüşmesi ölçüyor, yani
  regex katmanından fark yaratmıyor. Sonraki adım net: örnek cümle seti + kosinüs
  benzerliğiyle regex'ten önce bir ön-katman.
