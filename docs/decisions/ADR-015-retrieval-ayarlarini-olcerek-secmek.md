# ADR-015: Retrieval ayarlarını ölçerek seçmek, ayrı bir doğrulama setiyle

## Bağlam
Hibrit retriever'ın üç ayarı da bakıldığında makul görünüyordu ve hiçbiri
ölçülerek seçilmemişti: aday havuzu 8, harman 0.5 vektör + 0.5 BM25, BM25
tokenizasyonu boşlukla bölme.

Sohbet üzerinde çalışırken çıkan şey: "çalışma saatleriniz" sorgusunda en
üstteki doküman **Hesap İşletim Ücretleri**, "kartımı nasıl bloke ederim"
sorgusunda **Çalışma Saatleri** geliyordu. Ham skorlara bakınca sebep netleşti —
anahtarsız demo backend'inde (`FakeHashEmbeddings`) ilgisiz bir sorgu ("futbol
maçı sonucu", 0.157) ilgili bir sorgudan ("kartımı nasıl bloke ederim", 0.115)
yüksek skor alıyor. Vektör kanalı bu yapılandırmada bilgi taşımıyor.

Demo kısıtı olarak kabul edilebilirdi. Kabul edilemez olan görünmez olması:
BM25 doğru cevabı çoğu zaman kurtardığı için sistem çalışıyor gibi duruyordu.

## Seçenekler
- **A: Bırak, README'ye not düş.** Ucuz, ama sonraki değişikliğin iyileştirme mi
  bozma mı olduğu yine bilinemez.
- **B: Bir sorgu seti yaz, en iyi sayıları seç.**
- **C: Ayar seti + hiç dokunulmayan ayrı bir doğrulama seti.**

## Tercih
**C** — ve bu ADR'nin asıl sebebi B'nin neden yetmediği.

18 sorgulu bir set yazıp parametre tarandı. `havuz=20, vektör payı=0.25,
gövde=5` bu sette **%66.7 → %88.9** yaptı; ikna edici bir sayı. Sonra hiç
kullanılmamış ikinci bir 18 sorgu yazıldı. Aynı yapılandırma orada **%50.0 →
%50.0**. Kazanç sahteydi: sayılar retrieval'a değil, sete uydurulmuştu.

Izgara ikinci set de dahil edilerek yeniden tarandığında dayanan tek
yapılandırma kaldı:

| yapılandırma | ayar seti | doğrulama seti |
|---|---|---|
| havuz=8, vektör payı=0.50, gövde yok (önceki) | %66.7 | %50.0 |
| havuz=24, vektör payı=0.25, gövde=5 (şimdiki) | %88.9 | %61.1 |

Doğrulama setindeki 11.1 puan, ayarlarken o sete hiç bakılmadığı için gerçek.
İki set de 18 sorgu: tek sorgu ±%5.6, yani bu sayılar kesin değil, yön gösteriyor.

Üç değişiklik, üç ayrı sebep:

1. **Aday havuzu 8 → 24.** Performans ayarı gibi görünüyor ama değil: BM25
   yalnızca havuzun içini sıralayabiliyor, yani havuz aynı zamanda sözcüksel
   kanalın görebileceği en geniş küme. Dar havuz, "hibrit" mimariyi sessizce
   "zayıf embedding ne derse o"ya çeviriyordu.
2. **Vektör payı 0.5 → 0.25.** Her havuz boyutunda 0.5 ölçülen en kötü değerdi.
   Sıfır da olabilirdi ama o zaman bu hibrit olmaktan çıkardı; 0.25 kanalı açık
   tutuyor ve ölçümde 0.0 kadar iyi.
3. **BM25 tokenizasyonuna Türkçe gövdeleme.** "şifremi" ile "şifre" boşlukla
   bölen bir BM25 için iki ayrı terim. Sabit uzunlukta önek kesme (5 harf) kaba
   ama bağımlılık eklemiyor; 4 ve 6 ölçüldü, ikisi de daha kötü.

## Sonuçları
- `rag_candidate_pool` ve `rag_vector_weight` artık ayar, gömülü sabit değil.
  Gerçek bir embedding sağlayıcısına geçildiğinde **yeniden ölçülmeli** —
  mevcut değerler yalnızca `FakeHashEmbeddings` için ölçüldü.
- `python -m evaluation.retrieval` bölüm bazlı top-1 basıyor, `--sweep`
  ızgarayı tarıyor. Karar tekrarlanabilir durumda.
- Havuz genişleyince kuyrukta negatif cosine skorları çıkıyor ve Chroma
  uyarıyor. Beklenen bir durum, çağrı yerinde gerekçesiyle susturuldu. Negatif
  skorlu adaylar elenmiyor: elemek tam da düzeltilen hatayı geri getirirdi.

## Bilerek yapılmayan
Tavan embedding'de, reranker'da değil — doğrulama setinde %61.1, %90 değil.
Bunu kapatmanın yolu gerçek bir embedding modeli, parametre ayarı değil. Sayıyı
daha fazla zorlamak, ikinci seti de birinciye çevirmek olurdu.

Alaka eşiği (düşük skorlu sonucu hiç göstermemek) denenmedi. Gerekçesi var:
bilgi tabanında olmayan bir konuda ("havale ücreti") sistem şu an en yakın
dokümanı kaynak göstererek güvenle cevap veriyor. Ama eşik de ölçülerek
seçilmesi gereken bir sayı; ayrı bir karar olarak duruyor.
