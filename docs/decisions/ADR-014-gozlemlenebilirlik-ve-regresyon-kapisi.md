# ADR-014: Gözlemlenebilirlik (Langfuse) ve regresyon kapısı

**Durum:** Kabul edildi
**Tarih:** 2026-09-21

## Bağlam

Bu sistem üretimde çalışırken şu sorunla karşılaşıldı: cevaplar bozulmaya
başladığında ilk refleks modeli suçlamak oluyordu. Gerçek sebep çoğu zaman
başka bir katmandaydı — yönlendirme yanlış worker'a gidiyordu, retrieval
alakasız chunk döndürüyordu, ya da prompt'a giren bağlam eksikti. Model doğru
akıl yürütüyordu; yanlış olan ona ulaşan veriydi.

Trace olmadan bu ayrım yapılamıyor. Elde yalnızca "kullanıcı yanlış cevap
aldı" bilgisi varken, prompt değiştirmek en kolay ama en az bilgilendirici
hamle: bazen düzeliyor, neden düzeldiği bilinmiyor, ve bir sonraki bozulmada
aynı tahmin döngüsü baştan başlıyor.

İkinci sorun regresyon. Mevcut `eval_harness` doğruluk ölçüyordu ama hiçbir
şeyi **engellemiyordu**. Bir prompt değişikliğinin retrieval'ı bozması ancak
biri elle çalıştırıp sayıya bakarsa fark ediliyordu.

## Karar

**1. Grafik çağrısının tamamı Langfuse ile trace'lenir.**

Callback handler, `build_trace_config()` üzerinden `graph.ainvoke`'a veriliyor.
Bu sayede her düğüm — NER, intent, supervisor, worker'lar, guardrail — tek bir
trace ağacında görünüyor ve bir cevabın hangi yoldan geçtiği tahmin değil kayıt
oluyor.

`conversation_id` Langfuse oturumu olarak geçiyor. Bir hatayı incelerken
ihtiyaç duyulan ilk şey, o turn'ün tek başına değil konuşmanın geri kalanıyla
birlikte okunması.

**2. Tracing eksik yapılandırmada sessizce kapanır.**

`LLM_PROVIDER=fake` ile aynı felsefe: anahtar yoksa, paket kurulu değilse ya da
Langfuse erişilemiyorsa sistem aynen çalışır, `get_trace_callbacks()` boş liste
döner. Gözlemlenebilirlik bir teşhis aracı, bir çalışma önkoşulu değil. `pip
install -e ".[tracing]"` bilinçli olarak opsiyonel ekstra.

**3. Örnekleme deterministik.**

`langfuse_sample_rate < 1.0` olduğunda seçim `conversation_id` hash'ine göre
yapılıyor, rastgele değil. Rastgele seçim bir konuşmanın turn'lerinin yarısını
kaybettirir ve geriye okunamayan bir trace bırakır.

**4. Değerlendirme bir kapıya bağlandı.**

`evaluation/report.py` doğruluğun yanında **gecikme** de ölçüyor (p50/p95) ve
sonucu `data/eval_baseline.json` ile karşılaştırıyor. `make eval-gate`
regresyonda exit 1 veriyor, yani CI'da build'i kırıyor.

Toleranslar sıfır değil: doğrulukta ±0.05, gecikmede 1.5×. Embedding
sağlayıcısı kaynaklı küçük oynamalarda CI kırmak, kapının kapatılmasıyla
sonuçlanır — yanlış alarm veren bir kapı, olmayan bir kapıdır.

## Alternatifler ve neden seçilmedi

**RAGAS / LLM-as-judge.** Değerlendirmenin kendisini non-deterministik yapar.
Aynı kod iki koşuda iki farklı skor üretirse regresyon kapısı gürültüye boğulur
ve güvenilmez hale gelir. Buradaki her metrik deterministik olmak zorunda.

**LangSmith.** LangChain ekosistemine daha sıkı entegre ama kapalı kaynak ve
kendi sunucusunda çalıştırılamıyor. Bu proje bankacılık alanını modelliyor;
konuşma içeriğinin üçüncü taraf bir SaaS'a gitmesi bu alanda gerçek bir
kısıt. Langfuse self-host edilebiliyor.

**OpenTelemetry ile elle enstrümantasyon.** Daha genel ama LLM'e özgü olan
şeyi — token sayımı, maliyet, prompt/completion çiftleri — kendimiz yazmamız
gerekirdi. Kazanç, maliyeti karşılamıyor.

## Sonuçlar

- Bir cevap bozulduğunda hangi katmanın sorumlu olduğu bakılarak öğreniliyor.
- Prompt ve retrieval değişiklikleri artık ölçülmeden merge edilemiyor.
- Ek bağımlılık opsiyonel; CI ve offline çalıştırma etkilenmiyor.
- Bedeli: her istekte bir callback handler kurulumu ve ağ üzerinden batch
  gönderim. Örnekleme oranı bunun için var.
