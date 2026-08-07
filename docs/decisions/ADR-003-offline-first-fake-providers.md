# ADR-003: "Fake" LLM/embedding sağlayıcıları ile offline-first tasarım

## Bağlam
Bu bir portföy/demo projesi: gerçek bir API anahtarı olmadan da (CI dahil) uçtan uca
çalışması ve test edilebilmesi gerekiyor. Aynı zamanda gerçek bir sağlayıcıya (Anthropic/
OpenAI) tek satırlık bir config değişikliğiyle geçebilmeli.

## Seçenekler
- **A: Sadece gerçek sağlayıcı** — basit ama API anahtarı yoksa hiçbir şey çalışmaz,
  CI'da her PR gerçek para harcar ve ağ bağımlılığı yüzünden testler kırılgan olur.
  Anahtar sızması riski.
- **B: Her yerde `unittest.mock.patch`** — testler geçer ama üretim kodu hâlâ "anahtarsız
  açılmıyor" durumunda kalır; demo/CV incelemesi yapan biri projeyi klonlayıp
  `docker compose up` dediğinde 500 hatasıyla karşılaşır.
- **C: Birinci sınıf `LLM_PROVIDER=fake` / `EMBEDDING_PROVIDER=fake` modu** — deterministik,
  hash tabanlı sahte model + embedding, gerçek arayüzlerle (`BaseChatModel`, `Embeddings`)
  aynı sözleşmeyi uygular.

## Tercih
**C.** `Settings.resolved_llm_provider()` anahtar yoksa sessizce `FAKE`'e düşüyor (config.py),
böylece hem `docker compose up` hem `pytest` hiçbir gizli anahtar olmadan gerçekten çalışıyor.
`FakeChatModel` ve `FakeHashEmbeddings` gerçek sınıflarla aynı arayüzü uyguladığı için worker
kodu "fake mi gerçek mi" diye dallanmak zorunda kalmıyor — tek istisna `tool_agent`'ın
`is_fake_model` kontrolüyle bind_tools yerine deterministik bir entity→tool eşlemesine
düşmesi (bkz. `agents/workers/tool_agent.py` docstring'i).

## Sonuçlar
- ✅ `git clone` → `docker compose up` → çalışan bir demo, sıfır konfigürasyon.
- ✅ CI hiçbir API anahtarı/ağ bağımlılığı olmadan gerçek kodu (mock değil, gerçek graph)
  çalıştırıyor.
- ❌ Fake modda RAG/intent/NER yanıt *kalitesi* gerçek bir modelinkinden çok daha zayıf;
  bu fark README'de ve `.env.example`'da belirtiliyor.
