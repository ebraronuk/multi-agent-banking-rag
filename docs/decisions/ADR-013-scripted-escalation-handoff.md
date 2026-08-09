# ADR-013: İnsana aktarımı script'li, çok-aşamalı bir akış yapma

## Bağlam

`escalate_node` (ADR-006'nın "guardrail bir LLM çağrısı olmamalı" ilkesiyle aynı ruhta,
LLM'siz) ESCALATE niyetinde tek, statik bir cümle döndürüyordu: "Sizi bir müşteri
temsilcisine aktarıyorum." Bundan sonra konuşma normal akışa dönüyordu — kullanıcı bir
sonraki mesajında sıfırdan yeniden sınıflandırılıyordu.

Bu, canlı testte üç ayrı sorun çıkardı:

1. Bankacılık kelime dağarcığı içermeyen bir takip mesajı ("aktarım yapıldı mı") OUT_OF_SCOPE'a
   düşüp az önce "sizi aktarıyorum" diyen bir asistanın hemen ardından "bu konu kapsam dışı"
   demesi gibi tutarsız bir deneyime yol açıyordu.
2. Daha ciddisi: aynı mesaj gerçek bir LLM tarafından TRANSACTION_ACTION sanılabiliyordu
   ("aktarım" kelimesi hem "insana aktarma" hem "para transferi" demek) — model escalate
   akışını tamamen atlayıp kendi kafasından bir yanıt uydurdu.
3. İlk script'li sürümde (aktarım → ayrı bir turda temsilci karşılaması → ayrı bir turda
   kimlik doğrulama isteği) kullanıcı aktarım istedikten hemen sonra şikayetini yazınca bu
   mesaj **hiçbir yerde kullanılmıyordu** — karşılama turu içeriğe bakmaksızın sabit bir
   "nasıl yardımcı olabilirim" döndürüyordu, kullanıcı şikayetini ikinci kez yazmak zorunda
   kaldı. Ayrıca doğrulama bittikten sonra "ne zaman dönüş yapacaksınız?" gibi son derece
   doğal bir takip sorusu script'in tamamen dışına düşüp genel "kapsam dışı" cevabına
   düşüyordu.

## Seçenekler

- **A: Tek statik mesaj olarak bırak.** Yukarıdaki #1/#2'ye açık.
- **B: Aktarımdan sonrasını bir LLM'e "temsilci gibi davran" diye ürettirmek.** Reddedildi:
  ADR-006/ADR-009'un zaten kaçındığı riski (modelin bankanın tutamayacağı bir vaadi ya da
  var olmayan bir işlemi doğaçlaması) buraya taşırdı.
- **C: Script'li, çok aşamalı bir state machine.** Aşama `agents/memory.py` üzerinden
  turlar arası taşınıyor (ADR-008'deki `carried_pending_request` ile aynı desen).

## Tercih

**C**, ama ilk denemeden (ayrı aktarım/karşılama/doğrulama turları) sonra canlı geri
bildirimle iki kez revize edildi. Son hâli:

**Aşamalar:** `None` (aktarılmamış) → `verifying` (aktarım + temsilci "Aylin" karşılaması
+ kimlik doğrulama isteği **TEK turda birleşik**) → `awaiting_issue` (doğrulama başarılı,
sorun **sadece burada, tek seferde** soruluyor) → `resolved` (sorun kaydedildi, somut bir
SLA verildi — "en geç 24 saat içinde") → `None`'a geri döner.

İki tasarım kararı doğrudan #3'teki geri bildirime cevap:

- **Aktarım + karşılama + doğrulama isteği neden tek turda?** Çünkü kullanıcı "bir
  temsilciyle görüşmek istiyorum" dedikten sonra ayrı bir "merhaba, nasıl yardımcı
  olabilirim" round-trip'i beklemek hem gereksiz hem kullanıcının bir sonraki mesajını
  (genelde şikayetinin kendisi) script'in görmezden gelmesine yol açıyordu.
- **Sorun neden doğrulamadan SONRA soruluyor?** Gerçek bir banka desteği de önce kimliği
  doğrular, sonra hesaba özel bir konuyu konuşur — bu hem daha gerçekçi hem de sorunun
  yalnızca BİR kez, doğrulama bitince net bir şekilde sorulmasını sağlıyor.
- **`resolved` aşaması neden var?** "Ne zaman dönüş yapacaksınız?" gibi bir kapanış-sonrası
  soruyu yakalayıp somut bir cevap (SLA'yı tekrar) vermek için — aksi halde script bitip
  normal sınıflandırmaya düşüyor, ki bu da genel "kapsam dışı" cevabına yol açıyordu. Bu
  aşama tek bir turluk: cevap verildikten sonra `None`'a dönüyor, konuşmayı sonsuza kadar
  script içinde tutmuyor.

`supervisor.py`'nin router'ı `carried_escalation_stage` `None` değilken, o turda
sınıflandırılan `intent` ne olursa olsun doğrudan `escalate_node`'a yönlendiriyor — script
bir kez başladıktan sonra sınıflandırıcının o turda ne düşündüğü önemli değil, #2'deki
hatayı yapısal olarak imkansız kılıyor.

Ayrıca `escalate_node`, script aktifken döndürdüğü `intent`'i her zaman `ESCALATE`'e
sabitliyor — o turun ham sınıflandırması (ör. OUT_OF_SCOPE) API'nin `intent` alanına hiç
sızmıyor. Bunun sebebi de canlıda görülen somut bir hata: frontend'in etiket satırı
(`INTENT_LABELS`) ham `intent`'i gösteriyordu, yani Aylin'in mesajının altında "KAPSAM
DIŞI" yazıyordu — cevabın içeriğiyle çelişen bir etiket.

`verifying` aşamasında kullanıcının mesajında 4 haneli bir sayı aranıyor (regex, NER'a ya
da `banking_repository`'ye karşı bir doğrulama değil) — bulunursa doğrulama başarılı
sayılıp `awaiting_issue`'ya geçilir, bulunamazsa aynı istek tekrarlanır. Arayüzde bunun
demo olduğu ve herhangi bir 4 haneli numaranın kabul edildiği açıkça belirtiliyor
(`ChatPanel.tsx`, chat dışında statik bir not — sohbetin kendisine, bir mesaj olarak
değil).

Gerçek bir insan hiçbir aşamada bağlanmıyor — "Aylin" script'li, sabit bir persona, tıpkı
diğer tüm mesajlar gibi önceden yazılmış, bir LLM'in o an doğaçladığı biri değil. Repliklerini
bir LLM üretseydi ADR-006'nın guardrail için reddettiği aynı risk (modelin ne söyleyeceğinin
tahmin edilememesi) burada da geçerli olurdu. Bu bir portföy projesi, gerçek bir müşteri
hizmetleri kuyruğu yok — amaç tam bir aktarım/doğrulama/çözüm UX akışının sahte bir vaat
üretmeden nasıl modelleneceğini göstermek.

## Sonuçlar

- ✅ "aktarım yapıldı mı" tarzı bir takip mesajı artık sınıflandırıcının o turdaki
  tahminine bakılmaksızın script'e devam ediyor.
- ✅ Kullanıcı şikayetini artık hiçbir zaman iki kez yazmak zorunda kalmıyor — sorun tam
  olarak bir kez, doğrulamadan hemen sonra soruluyor.
- ✅ "Ne zaman dönüş yapacaksınız?" gibi doğal bir kapanış-sonrası soru artık genel
  "kapsam dışı" cevabına düşmüyor, somut bir süre alıyor.
- ✅ API'nin raporladığı `intent`, script aktifken tutarlı bir şekilde `ESCALATE` — arayüz
  etiketi artık cevabın içeriğiyle çelişmiyor.
- ✅ Script LLM'siz kaldığı için (ADR-006/ADR-009 ile aynı ilke) hiçbir aşamada bir modelin
  tutulamayacak bir vaat uydurma riski yok.
- ✅ (sonradan eklendi) `awaiting_issue` aşaması ilk halinde her mesajı kör kör bir şikayet
  sayıp kaydediyordu — canlıda "EFT limitim ne kadar?" gibi anında cevaplanabilir bir soruya
  bile anlamsızca "bu konuyu kaydettim, 24 saat içinde dönüş yapacağız" diyordu.
  `escalate_node` LLM'siz olduğu için bunu gerçekten cevaplayamıyor ama `rag_agent`
  cevaplayabiliyor — `supervisor.py`'nin router'ı artık bu tek durumda (`awaiting_issue` +
  RAG_QUERY) script'i atlayıp gerçekten cevaplıyor, `verifying` aşamasında bu istisna yok
  (kimlik doğrulanmadan hiçbir soru script'i atlayamıyor).
- ❌ Script bir kez başladıktan sonra "vazgeç"/"iptal et" gibi bir çıkış yolu yok —
  kullanıcı doğrulama adımına gelene kadar başka bir bankacılık isteği soramıyor. Bilinçli
  bir basitleştirme: gerçek bir destek akışı da genelde önce doğrulamayı ister.
- ❌ Doğrulama gerçek değil — herhangi bir 4 haneli sayı kabul ediliyor,
  `banking_repository`'deki gerçek müşteri/kart verisine karşı kontrol edilmiyor. Bu bir
  UX akışı demosu, bir kimlik doğrulama sistemi değil; arayüzde açıkça belirtiliyor.
- ❌ `resolved` aşamasının kapanışı da tek turluk — kullanıcı "ne zaman"ı iki kez üst üste
  sorarsa ikinci soru artık script dışında (normal sınıflandırmaya düşer). Kabul edilebilir:
  gerçek bir kapanıştan sonra art arda aynı soru beklenen bir senaryo değil.
