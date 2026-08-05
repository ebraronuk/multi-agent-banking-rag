# ADR-007: Hata taksonomisi HTTP seviyesiyle sınırlı; ajan seviyesi hatalar 200 + flag

## Bağlam
`schemas/dto.py::ErrorCode` başlangıçta 6 üye içeriyordu (`AGENT_ITERATION_LIMIT`,
`TOOL_EXECUTION_FAILED`, `GUARDRAIL_BLOCKED` dahil), ama kodun geri kalanı bunların
3'ünü **hiç raise etmiyordu** — iterasyon sınırı, guardrail engeli ve araç hatası
zaten `guardrail_agent.py` tarafından 200 + `guardrail_flags` olarak ele alınıyordu.
Ayrıca `rag_agent`/`smalltalk_agent`/`tool_agent` üçü de LLM çağrısını try/except'siz
yapıyordu — gerçek bir sağlayıcı kesintisi (rate limit, timeout, 5xx) doğrudan
`/chat`'i 500'e düşürürdü.

## Seçenekler
- **A: Her olası "aksama" için ayrı bir HTTP hata kodu tanımlamak** (kapsamlı ama
  koddaki gerçek davranışla senkron tutulması zor; kullanılmayan enum üyesi
  kodun "iddia ettiği" ama yapmadığı bir şeyi temsil eder).
- **B: HTTP hata taksonomisini gerçekten *raise edilen* durumlarla sınırlamak**
  (`VALIDATION_ERROR`, `RATE_LIMITED`, `INTERNAL_ERROR`); ajan-seviyesi "beklenen"
  aksamaları (iterasyon sınırı, guardrail engeli, eksik varlık) başarılı bir
  `ChatResponse` + `GuardrailFlag`/mesaj olarak modellemeye devam etmek; LLM
  çağrılarını `safe_ainvoke` ile sarmalayıp sağlayıcı hatasını guardrail'in
  zaten var olan `NO_DRAFT_PRODUCED` yoluna düşürmek.

## Tercih
**B.** İki ayrı prensip birleşiyor:
1. *"Sonuç döndür, exception'a boğma"* — kullanıcının "kartımı bulamadım" demesi
   bir programlama hatası değil, beklenen bir konuşma sonucu; 200 + açıklayıcı
   mesaj + flag, 4xx/5xx'ten daha doğru bir temsil.
2. *Kullanılmayan bir enum üyesi, hiç üye olmamasından beter* — `ErrorCode`'da
   asla raise edilmeyen bir kod bırakmak, "bunu ele aldık" yanılsaması yaratır.
   `agents/workers/tool_agent.py`, `rag_agent.py`, `smalltalk_agent.py` artık
   `app/core/llm.py::safe_ainvoke` üzerinden çağrı yapıyor: başarısız bir LLM
   çağrısı `draft_answer=None` (veya `tool_agent` için deterministik bir özet)
   döndürüyor, guardrail bunu zaten yakalıyor — ayrı bir `UPSTREAM_LLM_ERROR`
   yoluna hiç gerek kalmıyor.

## Sonuçlar
- ✅ `ErrorCode`'daki her üye gerçekten en az bir yerde raise ediliyor —
  taksonomi kodun davranışıyla birebir.
- ✅ Bir Anthropic/OpenAI kesintisi artık `/chat`'i 500'e düşürmüyor; kullanıcı
  "şu an yanıt üretemedim" tarzı bir mesaj alıyor (guardrail'in mevcut
  `NO_DRAFT_FALLBACK_MESSAGE`'ı üzerinden).
- ✅ `tool_agent`, özetleme LLM çağrısı başarısız olsa bile aracın gerçek
  sonucunu (`_format_tool_outcome`) kaybetmiyor — sadece özetleme adımı
  atlanıyor, veri atılmıyor.
- ❌ Bu, "her hata HTTP status koduna yansır" gibi daha basit/klasik bir REST
  varsayımını kırıyor — API tüketen bir istemcinin başarı/başarısızlığı
  `status_code` yerine `guardrail_flags`/`intent` alanlarına bakarak da
  değerlendirmesi gerekiyor. Bu, `docs/architecture.md`'de ve README'nin API
  sözleşmesi bölümünde açıkça belirtiliyor.
