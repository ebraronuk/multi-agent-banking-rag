# ADR-002: Supervisor rotalaması deterministik, LLM tabanlı değil

## Bağlam
Çoklu ajan sistemlerinde yaygın bir tuzak: "hangi ajan çalışsın" kararını da bir LLM'e
sormak. Bu, sistemi uçtan uca "prompt söylediği kadar tahmin edilebilir" hale getirir.

## Seçenekler
- **A: LLM router** — bir LLM'e "bu mesaj için hangi ajanı çağırmalıyım?" diye sorup
  yanıtını parse etmek.
- **B: Deterministik router** — `intent` (zaten NER'den sonra çıkarılmış) üzerinden
  düz `if/elif` ile hedef düğümü seçmek; iterasyon sınırını da burada saymak.

## Tercih
**B.** `IntentLabel` zaten ayrı bir ajan (`intent_agent`) tarafından üretiliyor; supervisor'ın
tek işi bu etiketi bir düğüm adına çevirmek. Bunu bir LLM çağrısına dönüştürmek: (1) her
turn'e ekstra gecikme + maliyet ekler, (2) yeni bir hata modu açar (LLM yanlış ajanı seçebilir,
üstelik bunu debug etmek "neden bu prompt böyle yanıt verdi" sorusuna döner), (3) test
edilebilirliği düşürür — deterministik bir router, girdi/çıktı eşlemesiyle birim testle
tam kapsanabilirken LLM router'ın davranışı örnekleme gerektirir.

## Sonuçlar
- ✅ Rota kararı %100 birim test edilebilir (`tests/unit/test_supervisor.py`).
- ✅ Ekstra LLM çağrısı yok → daha düşük gecikme/maliyet.
- ✅ İterasyon sınırı (`max_agent_iterations`) net bir yerde, kaçırılması imkansız bir
  şekilde uygulanıyor.
- ❌ Yeni bir intent eklemek supervisor'ı da güncellemeyi gerektiriyor (routing tablosu
  merkezi ama elle senkron tutuluyor). Kapsam bu projede küçük (7 intent) olduğu için
  kabul edilebilir; çok daha büyük bir intent kümesinde config-tablosuna taşınabilir.
