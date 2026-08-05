# ADR-010: Bankacılık verisi için Postgres + offline in-memory fallback

## Bağlam
`mcp_server/tools/banking_tools.py` başından beri hesap/kart/işlem verisini bir Python
dict'inde (`_ACCOUNTS`) tutuyordu — hiç SQL yok, hiç gerçek şema yok. Tool-calling akışını
göstermek için yeterliydi, ama sistemin geri kalanı (Chroma, Redis) gibi bu katmanın da
gerçek bir kalıcı depolamaya karşı çalıştığını göstermek istiyorsak, bu eksik kalıyordu:
`get_balance`/`list_transactions`/`block_card` gibi işlemler gerçek bir bankada tam olarak
ilişkisel bir veritabanına gider — parametreli sorgu, foreign key, transaction.

## Seçenekler
- **A: Dict fixture'ı olduğu gibi bırak.** Değişiklik yok, ama tool katmanının SQL
  tarafı hiç sergilenmemiş oluyor.
- **B: Her zaman Postgres zorunlu kıl.** Gerçek SQL'i doğrudan gösterir, ama ADR-003'ün
  "offline-first" ilkesini bozar — CI ve `docker compose`'suz lokal geliştirme artık
  ayakta bir veritabanı ister, tıpkı LLM/embedding/hafıza katmanlarının kaçındığı şey.
- **C: `agents/memory.py`'deki Redis/in-memory ayrımıyla aynı şekli kullan — `BankingRepository`
  arayüzü, `DATABASE_URL` set edilmişse `PostgresBankingRepository`, edilmemişse
  `InMemoryBankingRepository`.**

## Tercih
**C.** `mcp_server/tools/banking_repository.py`, `get_banking_repository(settings)` ile
hangi implementasyonun kullanılacağına bir kere karar veriyor:

- `InMemoryBankingRepository` — `SEED_ACCOUNTS`'un kendi kopyası üzerinde çalışır. Her
  örnek kendi kopyasını tuttuğu için (`copy.deepcopy`), testler birbirinin `block_card`
  mutasyonunu görmüyor — eski `_ACCOUNTS` modül-seviyesi dict'in testler arası "fixture'ı
  geri al" temizliğine ihtiyaç duyan halinden daha temiz bir davranış.
- `PostgresBankingRepository` — `db/schema.sql`'deki `accounts`/`cards`/`transactions`
  tablolarına parametreli SQL ile gider. Havuz `__init__`'te açılamıyor
  (`asyncpg.create_pool` bir coroutine), o yüzden ilk sorguda, bir `asyncio.Lock` ile
  korunarak kuruluyor.

Her iki implementasyon da aynı `{"ok": ..., "data"/"error": ...}` zarfını dönüyor — bir
Postgres bağlantı hatası da tıpkı "hesap bulunamadı" gibi `{"ok": False, "error": ...}`
olarak geliyor, exception olarak değil. Bu, `tool_agent.py`'nin zaten beklediği sözleşme;
bir veritabanı kesintisi turn'ü 500'e düşürmek yerine "şu an bakamadım" cevabına
yumuşuyor.

`db/schema.sql` hem `docker-compose.yml`'deki `postgres` servisinin
`docker-entrypoint-initdb.d`'si üzerinden hem de `scripts/seed_postgres.py` ile Docker'sız
lokal kullanım için uygulanıyor — aynı seed verisi (`SEED_ACCOUNTS` ile birebir aynı iki
demo müşteri), hangi backend'e karşı yazılırsa yazılsın bir test aynı şeyi anlatıyor.

## Sonuçlar
- ✅ CI ve lokal geliştirme hâlâ sıfır ek altyapıyla ayakta kalıyor — `DATABASE_URL`
  set edilmediği sürece hiçbir şey Postgres beklemez.
- ✅ Gerçek dağıtımda (`docker compose up`) `get_balance`/`list_transactions`/`block_card`
  gerçek, parametreli SQL çalıştırır — `db/schema.sql`'deki foreign key'ler ve unique
  constraint gerçek bütünlük kısıtları.
- ❌ `open_support_ticket`'ın Postgres implementasyonu hiçbir tabloya yazmıyor (bkz. o
  metodun docstring'i) — bilinçli: gerçek bir dağıtımda bu ayrı bir destek/ticketing
  sistemine gider, burada bir tablo açmak o entegrasyonu taklit etmekten öteye geçmezdi.
- ❌ İki backend arası veri tutarlılığı elle korunuyor (`SEED_ACCOUNTS` ve
  `db/schema.sql`'in seed'i birbirinden bağımsız yazıldı) — bir migration aracı (Alembic
  vb.) bu demo'nun kapsamı dışında bırakıldı.
