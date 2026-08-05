# ADR-008: Kısa süreli konuşma hafızası + slot-doldurma sürekliliği

## Bağlam
İlk sürümde `ChatRequest.conversation_id` alanı vardı ama hiç kullanılmıyordu — her
`/chat` çağrısı `new_state()` ile sıfırdan başlıyordu. Gerçek bir kullanım senaryosu
bunu hemen ifşa ediyor: kullanıcı "kartımı blokla" der, sistem "hangi kartın son 4
hanesi?" diye sorar, kullanıcı "4321" yazar — bağlamsız okunduğunda "4321" hiçbir şey
ifade etmiyor ve `nlp/ner_extractor.py`'nin `kart` anahtar kelimesine anchor'lı
regex'i bunu bir `CARD_LAST4` olarak yakalamıyor. Konuşma hafızası olmadan bu turn
`OUT_OF_SCOPE`'a düşüyor — kullanıcı isteğini en baştan tekrar yazmak zorunda kalıyor.

## Seçenekler
- **A: Hafıza yok, `conversation_id` sadece log korelasyonu için kalsın.** En basit,
  ama yukarıdaki senaryo hiç çözülmüyor — gerçek bir sohbet asistanı için ciddi bir eksik.
- **B: Tüm mesaj geçmişini LLM'e ham olarak geçirmek, ekstra bir mekanizma kurmadan.**
  Serbest metinli takip sorularını (RAG/sohbet) doğal şekilde çözer, ama "4321" gibi bir
  slot-doldurma cevabını LLM'in doğru intent'e bağlaması garanti değil — deterministik
  tool-agent yolu (ADR-002) bunu hiç görmez, çünkü intent/entity çıkarımı LLM'den önce,
  ayrı bir düğümde oluyor.
- **C: Redis'te (veya in-memory fallback) tur geçmişi + açık bir "ne bekleniyordu" alanı
  (`PendingEntityRequest`) tutmak; `memory_agent` bunu her turn'ün başında yükleyip
  `ner_agent`/`intent_agent`'ın bunu kullanmasını sağlamak.**

## Tercih
**C.** İki ayrı mekanizma birlikte çalışıyor:
1. **Genel geçmiş** (`state["history"]`) — `rag_agent`/`smalltalk_agent`/`tool_agent`'ın
   LLM çağrılarına ekleniyor (`agents/memory.py::history_to_messages`), serbest metinli
   takip sorularının doğal okunmasını sağlıyor ("ya EFT için mi?").
2. **Slot-doldurma sürekliliği** (`PendingEntityRequest`) — `tool_agent`, eksik bir varlık
   yüzünden turn'ü kısa devre yaptırdığında bunu açıkça kaydediyor (hangi intent, hangi
   varlık tipi bekleniyor). Bir sonraki turn'de `memory_load` bunu
   `state["carried_pending_request"]`'e yüklüyor; `ner_agent` bunu görüp bariz bir cevap
   (`synthesize_bare_answer_entity`: tam 4 haneli sayı, ya da bir IBAN) varsa sentezliyor;
   `intent_agent` da varlık gerçekten bulunduysa niyeti yeniden sınıflandırmak yerine
   kaydedilmiş niyeti aynen kullanıyor (`intent_confidence=1.0`, "slot-fill answered" trace'i).

`pending_entity_request` (bu turn'ün çıkışı) ve `carried_pending_request` (önceki turn'den
yüklenen giriş) bilinçli olarak **ayrı** state alanları — biri diğerine last-write-wins
ile karışırsa, konuyu değiştiren bir turn'den sonra eski bir slot-doldurma isteği
sessizce "askıda" kalabilir. `tool_agent` her turn'de `pending_entity_request`'i ya
açıkça yeniden set eder ya da hiç dokunmaz (varsayılan `None`), böylece konuyla
alakasız bir turn eski isteği otomatik olarak "unutur".

Redis + in-memory fallback seçimi ADR-003'ün "offline-first" felsefesiyle birebir aynı:
`REDIS_URL` yoksa (`agents/memory.py::get_conversation_memory`) process-içi bir dict'e
düşer — sıfır konfigürasyonla `docker compose up` ve testler hâlâ çalışır, sadece
restart'ta hafıza kaybolur ve replikalar arası paylaşılmaz. Prod'da Redis bunu çözer.

## Sonuçlar
- ✅ "Eksik bilgi sor → kullanıcı cevaplar → işlem tamamlanır" akışı gerçekten çalışıyor
  (`tests/integration/test_chat_api.py::test_multi_turn_slot_fill_...`, Docker'da canlı
  doğrulandı).
- ✅ Redis'e ya da hafızaya yazma/okuma hatası hiçbir zaman turn'ü 500'e düşürmüyor —
  `RedisMemory.load`/`save_turn` hatayı yutup boş bağlamla devam ediyor (bkz.
  `tests/unit/test_memory.py`'deki bağlantı-hatası testleri).
- ❌ `synthesize_bare_answer_entity` kasıtlı olarak dar (sadece tam 4 haneli sayı / IBAN
  deseni) — "muhtemelen bu cevaptır" gibi daha geniş bir tahmin, yanlış bir bankacılık
  işlemini tetikleme riskini artırırdı. Daha genel bir slot-doldurma (serbest metinli
  cevaplar) bu demo'nun kapsamı dışında bırakıldı.
- ❌ Redis'te kalıcılık yok (persistence volume yok, bkz. `docker-compose.yml` yorumu) —
  bilinçli: bu bir konuşma önbelleği, bir veritabanı değil; TTL (`CONVERSATION_TTL_SECONDS`)
  zaten unutmayı bekliyor.
