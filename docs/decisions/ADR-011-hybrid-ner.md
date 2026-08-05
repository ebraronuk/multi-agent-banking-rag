# ADR-011: Regex + LLM hibrit varlık çıkarımı (NER)

## Bağlam
`nlp/ner_extractor.py` baştan beri tamamen regex tabanlıydı: IBAN, tutar, tarih, kart
son 4 hane, hesap tipi — hepsi sabit desenlerle yakalanıyor. Bu, niyet sınıflandırmasıyla
(`nlp/intent_classifier.py`) asimetrik bir durumdu: `classify_intent` hem kural tabanlı
hem gerçek bir model bağlıyken LLM tabanlı bir yol sunuyordu (`with_structured_output` +
kural tabanlı fallback), ama NER tarafında sadece regex vardı. Regex'in kör olduğu somut
bir örnek: serbest metin kişi adları (`PERSON_NAME` tipi `schemas/dto.py`'de tanımlı ama
hiçbir regex bunu hiç üretmiyordu) ve regex'in kalıbına uymayan alışılmadık ifadeler
("geçen ayın on beşinde" gibi bir tarih).

## Seçenekler
- **A: Regex'i olduğu gibi bırak.** Basit ve hızlı, ama `PERSON_NAME` gibi bir tip
  şemada tanımlı olup hiçbir zaman üretilmiyor — ölü kod gibi duruyor.
- **B: Regex'i tamamen LLM ile değiştir.** Serbest metin recall'ı iyileşir, ama IBAN/kart
  son 4 hane gibi kesin, tek-doğru-cevaplı alanlarda regex'in verdiği %100 kesinliği ve
  sıfır-latency/sıfır-maliyet avantajını kaybettirir — özellikle `LLM_PROVIDER=fake`
  modunda (CI, testler) hiçbir varlık çıkmaz hale gelir.
- **C: Regex her zaman çalışır (taban), gerçek bir model bağlıysa üstüne bir LLM geçişi
  eklenir ve regex'in bulamadıkları birleştirilir.**

## Tercih
**C.** `extract_entities_with_llm(text, llm)`:
1. Regex geçişini (`extract_entities`) her zaman çalıştırır — bu hâlâ taban: hızlı,
   denetlenebilir, `LLM_PROVIDER=fake`'te (CI, testler) tek başına yeterli.
2. `is_fake_model(llm)` ise burada durur — bir hash'e "bu metinde hangi varlıklar var"
   diye sormanın anlamı yok, `classify_intent`'teki aynı kısa devre.
3. Gerçek bir modelde `llm.with_structured_output(_NERExtraction)` ile ikinci bir geçiş
   yapar (`agents/prompts/ner_prompt.py`), regex'in zaten bulduklarını (tip + normalize
   değer eşleşmesiyle) tekrar saymadan, sadece yenilerini (özellikle `PERSON_NAME`) ekler.
   LLM çağrısı başarısız olursa ya da beklenmeyen bir tip dönerse regex sonucuna düşer —
   `classify_intent`'in try/except/fallback şekliyle birebir aynı.

LLM'den gelen varlıklara `confidence=0.75` veriliyor (regex'in her zaman verdiği `1.0`'dan
düşük) ve `start`/`end` hiç set edilmiyor — LLM karakter offset'i veremiyor, bunu
uydurmak yanıltıcı olurdu. Birleştirilmiş liste sıralanırken regex'in bulduğu (pozisyonu
bilinen) varlıklar önce, LLM'in eklediği (pozisyonsuz) varlıklar sonra geliyor.

## Sonuçlar
- ✅ `PERSON_NAME` artık gerçekten üretilebilen bir tip — regex'te hiç yoktu.
- ✅ CI/testler hâlâ sıfır LLM çağrısıyla, sadece regex'le çalışıyor
  (`test_extract_entities_with_llm_uses_only_regex_for_fake_model`).
- ✅ Regex'in zaten kesin bulduğu bir varlığı LLM tekrar rapor etse bile, dedup mantığı
  (tip + normalize değer) onu ikinci kez eklemiyor
  (`test_extract_entities_with_llm_does_not_duplicate_entity_already_found_by_regex`).
- ❌ Dedup, tip + normalize-değer eşleşmesine dayanıyor — regex'in bulduğu bir IBAN'ı LLM
  hafifçe farklı bir yazımla (ör. boşluklu) rapor ederse normalize adımı ikisini de aynı
  forma indirgemediği sürece bir kopya sızabilir. Bu demo'nun ölçeğinde gözlemlenmedi,
  ama daha büyük bir varlık kümesinde ele alınması gereken bir sınır durumu.
- ❌ `intent_classifier.py`'deki gibi burada da LLM'in kendi hata payı var — yanlış bir
  varlık uydurması mümkün (ör. olmayan bir kişi adı). Bu varlıklar doğrudan bir bankacılık
  işlemini tetiklemiyor (o kısım `tool_agent.py`'nin argüman-doğrulama kontrolünden geçiyor,
  bkz. ADR-009) — NER'in kendisi işlem yapmıyor, sadece niyet sınıflandırmasına sinyal veriyor.
