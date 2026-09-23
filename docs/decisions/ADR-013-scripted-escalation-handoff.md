# ADR-013: İnsana aktarımı script'li, çok-aşamalı bir akış yapma

## Bağlam
`escalate_node` tek statik bir cümle döndürüyordu ("Sizi bir temsilciye aktarıyorum"),
sonra konuşma normal akışa dönüyordu. Canlıda üç şey patladı:

1. **Takip mesajı kapsam dışına düşüyordu.** "Aktarım yapıldı mı" bankacılık kelimesi
   içermediği için OUT_OF_SCOPE oluyordu — az önce "aktarıyorum" diyen asistan hemen
   ardından "bu konu kapsam dışı" diyordu.
2. **"Aktarım" iki anlamlı.** Gerçek LLM aynı mesajı TRANSACTION_ACTION sandı (insana
   aktarma / para transferi), escalate akışını atlayıp kendi cevabını uydurdu.
3. **Kullanıcı şikayetini iki kez yazmak zorunda kaldı.** İlk script'li sürümde karşılama
   turu içeriğe bakmadan sabit "nasıl yardımcı olabilirim" döndürüyordu; kullanıcının o
   turda yazdığı şikayet hiçbir yerde kullanılmıyordu.

## Seçenekler
- **A: Tek statik mesaj.** #1 ve #2'ye açık.
- **B: Aktarım sonrasını LLM'e "temsilci gibi davran" diye ürettirmek.** ADR-006/ADR-009'un
  kaçındığı riski buraya taşır: modelin bankanın tutamayacağı bir vaat doğaçlaması.
- **C: Script'li state machine.** Aşama `agents/memory.py` üzerinden turlar arası taşınıyor
  (ADR-008'deki `carried_pending_request` deseni).

## Tercih
**C.** İlk denemeden sonra canlı geri bildirimle iki kez revize edildi.

`None` → `verifying` (aktarım + "Aylin" karşılaması + kimlik doğrulama isteği, **tek
turda**) → `awaiting_issue` (sorun **sadece burada, bir kez** soruluyor) → `resolved`
(SLA verildi) → `None`.

- **Neden tek tur?** Ayrı bir karşılama round-trip'i, kullanıcının o turda yazdığı
  şikayeti script'e görünmez kılıyordu (#3).
- **Sorun neden doğrulamadan sonra?** Gerçek banka desteği de önce kimliği doğrular.
  Sorunun tam olarak bir kez sorulmasını da bu sağlıyor.
- **`resolved` neden var?** "Ne zaman dönüş yapacaksınız?" script bitince genel "kapsam
  dışı" cevabına düşüyordu. Tek turluk: cevaptan sonra `None`'a döner.

`supervisor.py` router'ı, `carried_escalation_stage` doluyken o turun `intent`'i ne olursa
olsun `escalate_node`'a gidiyor — #2'yi yapısal olarak imkansız kılıyor. `escalate_node`
ayrıca dönen `intent`'i `ESCALATE`'e sabitliyor; ham sınıflandırma (ör. OUT_OF_SCOPE)
API'ye sızmıyor. Bu da canlıdan geldi: frontend etiketi Aylin'in mesajının altında
"KAPSAM DIŞI" yazıyordu.

"Aylin" script'li sabit bir persona, LLM'in doğaçladığı biri değil. Doğrulama 4 haneli bir
sayı regex'i — `banking_repository`'ye karşı gerçek bir kontrol değil, arayüzde demo
olduğu açıkça yazıyor.

## Sonuçlar
- ✅ Takip mesajı sınıflandırıcının o turdaki tahminine bakılmaksızın script'te kalıyor.
- ✅ Şikayet bir kez soruluyor, kapanış sorusu somut süre alıyor.
- ✅ API'nin `intent`'i script boyunca tutarlı; arayüz etiketi cevapla çelişmiyor.
- ✅ Script LLM'siz olduğu için tutulamayacak vaat riski yok.
- ✅ (sonradan) `awaiting_issue` her mesajı kör kör şikayet sayıyordu; "EFT limitim ne
  kadar?" sorusuna bile "kaydettim, 24 saat içinde döneceğiz" diyordu. Router artık bu tek
  durumda (`awaiting_issue` + RAG_QUERY) script'i atlayıp cevaplıyor. `verifying`'de bu
  istisna yok — doğrulanmadan hiçbir soru script'i atlayamıyor.
- ✅ (sonradan) Çıkış yolu yoktu ve bu canlıda çıkmaza dönüştü: doğrulama
  adımındaki kullanıcı "boşver bakiyeme bakayım" yazınca "bunu bir doğrulama
  numarası olarak tanıyamadım" cevabını alıp akışta kilitli kalıyordu. "Gerçek
  destek akışları da önce doğrulama ister" gerekçesi yetersizdi — gerçek bir
  hatta da "vazgeçtim" demek mümkün. `verifying` ve `awaiting_issue`
  aşamalarında iptal artık kabul ediliyor.
- ❌ İptal ederken aynı cümlede yeni bir istek varsa ("boşver bakiyem ne
  kadar") aktarım kapanıyor ama ikinci istek kayboluyor — kart akışının
  aksine. Sebebi bu ADR'nin kendi garantisi: `carried_escalation_stage`
  doluyken router o turun niyetine bakmıyor ve bu, script'in ortasında yanlış
  worker'a düşme hatasını (#2) yapısal olarak imkansız kılıyor. Routing'i
  koşullu yapmak o garantiyi zayıflatır; kullanıcının isteğini tekrar
  yazması, script'in ortasında para transferi sanılmasından ucuz.
- ❌ Doğrulama gerçek değil, herhangi bir 4 hane kabul ediliyor. Bu bir UX akışı demosu.
- ❌ `resolved` tek turluk; aynı soru üst üste iki kez sorulursa ikincisi script dışında.
