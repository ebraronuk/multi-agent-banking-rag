# ADR-012: Tek mesajda birden fazla niyeti işleme (çoklu-niyet dispatch)

## Bağlam

`intent_agent`, bir mesaja her zaman tek bir `IntentLabel` atıyor; `supervisor` da bu tek
etikete göre tek bir worker'a yönlendiriyor. "Kartımı blokla ve EFT limitiniz ne kadar?"
gibi tek mesajda iki farklı kategoriye (CARD_ACTION + RAG_QUERY) giren bir istekte, sistem
sadece birini seçip diğerini hiç işlemiyor — kullanıcı cevabını göremeyen isteği ikinci bir
mesajla tekrar yazmak zorunda kalıyor.

Bu, tek etiketli niyet sınıflandırmasının bilinen bir sınırı: gerçek kullanıcı mesajları
genelde tek bir niyet taşır, ama bileşik istekler (özellikle sesli/serbest metin arayüzlerde)
az değil. `tool_agent`'ın kendi akıl yürütme döngüsü (ADR-009) zaten AYNI kategori içinde
birden fazla aracı bir turda çağırabiliyor ("kartımı blokla VE bir destek talebi aç") —
ama bu, supervisor'ın seçtiği TEK kategorinin içinde kalıyor; RAG_QUERY ile CARD_ACTION gibi
FARKLI kategoriler arasında bir köprü yok.

## Araştırma

- **LangGraph supervisor vs. swarm desenleri**: supervisor deseninde her etkileşim bir
  yönlendirme LLM çağrısından geçiyor (basit istekler için bile), swarm'da ilk ajan mesajı
  alıyor, karşılayamıyorsa doğrudan uygun uzmana devrediyor, supervisor'a geri dönmeden.
- **Anthropic'in kendi multi-agent araştırma sistemi**: bir lead agent planlıyor, birden
  fazla alt-agent'ı paralel çalıştırıyor, sonra bulguları ayrı bir sentez adımıyla
  birleştiriyor — "orchestrator-worker" deseni. Anthropic'in kendi ölçümünde bu, tek-ajanlı
  bir kuruluma göre ciddi bir kalite farkı yaratıyor, ama token maliyeti de yaklaşık 15 kat.
- **Kendi "Asisimo" projem** (aile asistanı, ~200K satır, 21 uzman agent, LangGraph):
  supervisor → orchestrator → uzman agent zincirinde, orchestrator katmanı zaten "tek
  mesajda birden fazla agent'ı sırayla çalıştıran çoklu-niyet dispatch" yapıyor
  (`orchestrator.node.js:88-209`), üstüne güven skoru yüksek + düşük riskli niyetlerde
  orchestrator'ı atlayan bir fast-path var. Bu depoda çözülmesi gereken problem, o projede
  zaten üretimde çalışan bir çözümle karşılanmış durumda — burada aynı fikri, bu projenin
  ölçeğine (4 kategori, tek turlu bileşik istekler) uyarlıyorum.

## Seçenekler

- **A: Değiştirme, tek niyet kalsın.** En basit, ama yukarıdaki senaryo çözülmeden kalıyor.
- **B: LangGraph'in `Send` API'siyle dinamik paralel fan-out.** Birden fazla worker'ı aynı
  anda tetikleyip sonuçları map-reduce ile birleştirmek mümkün — ama paralel çalışan
  worker'lar arasında `iteration_count`/`tool_calls` gibi paylaşılan state alanlarının
  reducer'larla doğru birleşmesini garanti etmek, özellikle `tool_agent`'ın kendi iç
  döngüsüyle birlikte, gerçek bir karmaşıklık kaynağı.
- **C: Sıralı fan-out — mevcut worker düğümlerini değiştirmeden, supervisor'a bir "sıradaki
  niyet" kuyruğu ve döngüsü eklemek.** Her worker kendi işini bitirince supervisor'a döner;
  kuyrukta bekleyen bir niyet varsa oraya yönlendirilir; hepsi bitince bir `synthesizer`
  düğümü toplanan taslakları tek bir cevapta birleştirir.

## Tercih

**C.** Gerekçe: `tool_calls` zaten `operator.add` reducer'ıyla passlar arası birikiyor,
`rag_agent`/`smalltalk` içinde hiçbir değişiklik gerekmiyor (state'i tekrar aynı düğümden
geçiriyoruz), ve tek-niyetli mesajlarda (istatistiksel olarak büyük çoğunluk) hiçbir ek LLM
çağrısı ya da graf adımı eklenmiyor — sadece `intent_agent`'ın zaten yaptığı structured-output
çağrısına bir opsiyonel alan (`extra_intents`) ekleniyor.

Akış: `intent_agent` (gerçek LLM'de) birincil niyetin yanında en fazla 2 farklı, ilgisiz
niyet daha döndürebiliyor. `supervisor`'ın router'ı, aktif niyetin worker'ı işini bitirince
(`worker_pass_done`/`tool_agent_done`) kuyrukta niyet kalıp kalmadığına bakıyor: kaldıysa
`advance_intent_node` (LLM'siz, sadece state geçişi) sıradakini aktif niyet yapıp worker'a
yönlendiriyor, bu turdaki taslak cevabı `collected_drafts`'a itiyor. Kuyruk boşalınca, birden
fazla taslak toplanmışsa `synthesizer` (gerçek LLM'de tek, doğal bir cevaba birleştirir; fake
modda taslakları numaralandırıp art arda ekler) devreye giriyor, değilse doğrudan guardrail'e
gidiliyor — bugünkü davranışla birebir aynı.

`ESCALATE`/`OUT_OF_SCOPE` zincire dahil edilmiyor: bir insana aktarım isteği genelde
konuşmanın o an bittiği anlamına geliyor, art arda başka bir işlem zincirlemek kafa
karıştırıcı olurdu. `SMALL_TALK` ise dahil — "EFT limitiniz ne kadar, bu arada merhaba"
gibi bir selamlamayı bir bilgi/işlem talebiyle birlikte taşımak gayet doğal.

## Sonuçlar

- ✅ "Kartımı blokla ve EFT limitiniz ne kadar?" gibi bileşik istekler tek turda, tek
  cevapta karşılanıyor — `tests/integration/test_chat_api.py::test_multi_intent_...`
- ✅ Tek-niyetli mesajlarda davranış ve maliyet birebir aynı kaldı — `extra_intents` boşsa
  `advance_intent`/`synthesizer` hiç çalışmıyor.
- ✅ `tool_calls` reducer'ı sayesinde iki farklı kategoriden gelen araç çağrıları
  (`block_card` + `get_balance`) tek `ChatResponse.tool_calls`'ta doğru birikiyor.
- ❌ İkinci niyetin işlenmesi, birincinin SONUCUNA bağlı olamıyor ("önce bakiyeme bak,
  düşükse bir uyarı ekle" gibi) — her pass birbirinden bağımsız çalışıyor, birinin çıktısı
  diğerinin girdisi olmuyor. Bu, README'nin daha önce de belirttiği "çok adımlı planlama"
  sınırıyla aynı yerde duruyor; çözümü ayrı bir ADR'yi hak eder.
- ❌ `max_agent_iterations` tüm turun genelinde tek bir sayaç — bileşik bir istek + her
  parçasında uzun bir `tool_agent` döngüsü aynı anda olursa, ikinci niyet hiç işlenmeden
  limite takılabilir. Demo ölçeğinde (varsayılan limit 6) gözlemlenmedi.
- ❌ Semantik (embedding tabanlı) bir niyet router katmanı eklenmedi — Asisimo'daki 3
  katmanlı (embedding → regex → LLM) yaklaşımın bir parçası, ve `rag/embeddings.py`
  altyapısı zaten var. Bilinçli olarak bu tura dahil edilmedi: varsayılan konfigürasyonda
  (`EMBEDDING_PROVIDER=fake`) `FakeHashEmbeddings` gerçek bir anlamsal benzerlik ölçmüyor,
  sadece token örtüşmesi ölçüyor — yani regex katmanından somut bir fark yaratmıyor,
  gerçek değeri ancak `EMBEDDING_PROVIDER=openai` ile ortaya çıkıyor. Sonraki adım olarak
  net: örnek cümle seti + kosinüs benzerliğiyle bir ön-katman, regex'ten önce.
