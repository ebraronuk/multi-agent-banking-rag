# ADR-001: LangGraph as the multi-agent orchestration layer

## Bağlam
Sistem, tek bir LLM çağrısıyla çözülemeyecek kadar farklı sorumlulukları (niyet/varlık
çıkarımı, bilgi tabanından yanıtlama, araç çağırma, güvenlik denetimi) ayrı adımlara
bölmeyi gerektiriyor. Bu adımlar arasında koşullu dallanma (intent'e göre farklı işçi),
döngü (araç çağırma birden fazla adım sürebilir) ve paylaşılan durum (state) gerekiyor.

## Seçenekler
- **A: Elle yazılmış fonksiyon zinciri** (`if/elif` ile sıralı çağrılar) — basit ama
  koşullu dallanma + döngü + durum yönetimi büyüdükçe okunaksızlaşır, iz (trace)
  üretmek için ayrı bir mekanizma gerekir.
- **B: LangGraph** — durumu (`GraphState`) düğümler arasında tip güvenli şekilde taşıyan,
  koşullu kenarları (conditional edges) ve döngüleri birinci sınıf destekleyen bir
  state-machine kütüphanesi. Grafiği tanımlamak zaten bir dokümantasyon formu.
- **C: CrewAI / AutoGen tarzı "ajan sohbeti" çerçeveleri** — ajanlar birbirleriyle
  serbest metin üzerinden konuşur; esnek ama iş kuralı (örn. "en fazla N iterasyon",
  "önce NER sonra intent") gizli/kırılgan hale gelir, davranışı test etmek zorlaşır.

## Tercih
**B: LangGraph.** İş akışı zaten bir durum makinesi gibi düşünülüyor (bkz. `agents/state.py`,
`agents/supervisor.py`); LangGraph bunu koddaki en doğal karşılığıyla ifade etmemizi
sağlıyor ve iş kuralı (routing) düz Python fonksiyonu olarak kalıyor (bkz. ADR-002).

## Sonuçlar
- ✅ Koşullu dallanma + döngü + paylaşılan durum framework tarafından yönetiliyor.
- ✅ Her düğüm bağımsız test edilebilir (girdi state → çıktı partial-state).
- ✅ `trace` alanı (append-only reducer) sayesinde her turn için "hangi düğüm ne yaptı"
  otomatik olarak API yanıtına yansıyor — ayrı bir logging şeması icat etmeye gerek yok.
- ❌ LangGraph'a bağımlılık: framework'ün kendi versiyon değişiklikleri (state reducer
  API'si gibi) upgrade maliyeti getirebilir. Kabul edilebilir; alternatiflerin hiçbiri
  bu kapsam için daha ucuz değildi.
