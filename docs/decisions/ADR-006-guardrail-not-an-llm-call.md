# ADR-006: Guardrail düğümü kural tabanlı, bir LLM çağrısı değil

## Bağlam
Grafiğin her yolu (RAG, araç çağırma, sohbet, eskalasyon) son adım olarak bir
"güvenlik/politika" denetiminden geçiyor: PII sızıntısı (IBAN/kart no gibi uzun sayı
dizileri), yatırım tavsiyesi dili, iterasyon sınırı aşımı.

## Seçenekler
- **A: Bu kontrolü de bir LLM'e yaptırmak** ("bu yanıt politika ihlali içeriyor mu?").
- **B: Deterministik regex/kural tabanlı kontrol** (`agents/workers/guardrail_agent.py`).

## Tercih
**B.** Bir guardrail'in tüm amacı, kontrol ettiği şeyin *dışında* ve *ondan bağımsız*
olmak. Yanıtı üreten aynı sınıf modele (hatta bazen aynı modele) "bunu sen mi kontrol
edeceksin" diye sormak, prompt injection'a karşı ekstra bir savunma katmanı sağlamıyor —
yanıtı manipüle edebilen bir girdi, kontrolü de manipüle edebilir. Regex tabanlı kontrol
daha az esnek (yeni bir PII deseni elle eklenmeli) ama davranışı %100 öngörülebilir ve
test edilebilir.

## Sonuçlar
- ✅ Guardrail'in davranışı LLM'in o anki yanıtından bağımsız, dolayısıyla jailbreak'e karşı
  daha dayanıklı bir son hat.
- ✅ Regex tabanlı redaksiyon (`_redact_sensitive_numbers`) birim testle tam kapsanabiliyor.
- ❌ Bilinmeyen/yeni bir PII deseni (ör. yeni bir ülkenin IBAN formatı) otomatik yakalanmaz,
  elle eklenmesi gerekir — bu projenin kapsamında (tek ülke, TR IBAN) kabul edilebilir bir
  sınır, ölçeklenirken not edilmesi gereken bir teknik borç.
