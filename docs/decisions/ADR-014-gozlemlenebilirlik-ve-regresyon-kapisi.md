# ADR-014: Langfuse tracing + regresyon kapısı

## Bağlam
Bir cevap bozulduğunda ilk refleks modeli suçlamak oluyordu. Gerçek sebep çoğu zaman
başka bir katmandaydı: yönlendirme yanlış worker'a gidiyor, retrieval alakasız chunk
döndürüyor, ya da prompt'a giren bağlam eksik. Trace olmadan bu ayrım yapılamıyor, elde
sadece "kullanıcı yanlış cevap aldı" kalıyor.

İkincisi, `eval_harness` doğruluk ölçüyordu ama hiçbir şeyi engellemiyordu.

## Seçenekler
- **A: LangSmith.** LangChain'e daha sıkı entegre ama kapalı kaynak, self-host yok.
- **B: OpenTelemetry + elle enstrümantasyon.** Token sayımı, maliyet, prompt/completion
  eşleştirmesini kendimiz yazmamız gerekirdi.
- **C: Langfuse.** Self-host edilebiliyor, LangChain callback'i hazır.

## Tercih
**C.** Bu proje bankacılık alanını modelliyor; konuşma içeriğinin kapalı bir SaaS'a
gitmesi bu alanda gerçek bir kısıt, LangSmith bu yüzden elendi. B'nin kazancı maliyetini
karşılamıyor.

Üç uygulama kararı:
- **Eksik yapılandırmada sessizce kapanır** (`LLM_PROVIDER=fake` ile aynı felsefe).
  `langfuse` opsiyonel ekstra — CI bir gözlem aracının varlığına bağlanmamalı.
- **`conversation_id` oturum olarak geçiyor.** Bir hatayı incelerken ihtiyaç duyulan ilk
  şey, o turn'ün konuşmanın geri kalanıyla birlikte okunması.
- **Örnekleme deterministik** (hash tabanlı). Rastgele seçim bir konuşmanın turn'lerinin
  yarısını kaybettirir, geriye okunamayan bir trace kalır.

Kapı tarafında: `evaluation/report.py` doğruluğun yanında p50/p95 ölçüyor ve sonucu
`data/eval_baseline.json` ile karşılaştırıyor. `make eval-gate` regresyonda exit 1.

**RAGAS ve LLM-as-judge bilinçli olarak dışarıda.** İkisi de değerlendirmenin kendisini
non-deterministik yapar; aynı kod iki koşuda iki skor üretirse kapı gürültüye boğulur ve
kapatılır.

## Sonuçlar
- ✅ Bozulmanın hangi katmandan geldiği bakılarak öğreniliyor, tahmin edilmiyor.
- ✅ Prompt ve retrieval değişiklikleri ölçülmeden merge edilemiyor.
- ❌ Toleranslar sıfır değil (doğrulukta ±0.05, gecikmede 1.5×). Sıfır tolerans,
  embedding sağlayıcısı kaynaklı oynamalarda CI'ı kırar ve kapının kapatılmasıyla
  sonuçlanır — yanlış alarm veren kapı, olmayan kapıdır. Bedeli: bu aralıktaki gerçek
  bir gerileme yakalanmaz.
- ❌ Her istekte bir callback handler kurulumu ve ağ üzerinden batch gönderim.
  `LANGFUSE_SAMPLE_RATE` bunun için var.
