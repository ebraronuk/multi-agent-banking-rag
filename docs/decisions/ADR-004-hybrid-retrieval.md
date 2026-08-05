# ADR-004: Hibrit retrieval (vektör + BM25) tek başına vektör aramaya karşı

## Bağlam
RAG ajanının bilgi tabanı küçük (birkaç düzine chunk), Türkçe, ve büyük ölçüde
IBAN/ücret/limit gibi kesin terimler içeriyor. Saf vektör arama, anlamsal olarak yakın
ama terim olarak yanlış sonuçları öne çıkarabiliyor (ör. "EFT limiti" sorgusu "havale
ücreti" chunk'ını "yakın anlamlı" diye getirebilir).

## Seçenekler
- **A: Sadece vektör arama (Chroma + embedding benzerliği).**
- **B: Sadece BM25 (terim frekansı).**
- **C: Hibrit — vektör + BM25 skorlarını normalize edip ağırlıklı ortalama.**

## Tercih
**C.** `rag/retriever.py` önce vektör aramadan aday kümesi çeker (`k_vector=8`), sonra
`rag/reranker.py` bu adaylar üzerinde BM25 skoru hesaplayıp iki skoru min-max normalize
ederek 0.5/0.5 birleştiriyor ve nihai `k_final=4` sonucu döndürüyor. Böylece hem anlamsal
yakınlık hem tam terim eşleşmesi (özellikle Türkçe finansal terimler için önemli) skora
katkı sağlıyor.

## Sonuçlar
- ✅ "EFT limiti nedir" gibi terim-ağırlıklı sorgularda daha isabetli sıralama.
- ✅ Chunk sayısı küçük olduğu için BM25'in ekstra maliyeti ihmal edilebilir düzeyde.
- ❌ Reranking adımı ekstra bir kütüphane (`rank-bm25`) ve birkaç satır normalize/birleştirme
  mantığı ekliyor — YAGNI sınırında; bilgi tabanı çok büyürse (binlerce chunk) BM25'i
  tüm koleksiyon yerine sadece vektör adayları üzerinde çalıştırmak bu maliyeti düşük tutuyor.

## Ölçülmüş bulgu: FAKE modda hibrit retrieval'in gerçek tavanı

`src/evaluation/eval_harness.py::run_retrieval_eval` 6 sorgu/beklenen-doküman çiftiyle
precision@1'i ölçüyor. `EMBEDDING_PROVIDER=fake` (yani `FakeHashEmbeddings`, anahtar
gerektirmeyen varsayılan) ile ölçülen sonuç **~%50** — varsayımla değil, çalıştırıp
ölçerek bulundu (`tests/unit/test_eval_harness.py`). İki bileşenli bir sınır:

1. `FakeHashEmbeddings` anlamsal benzerlik taşımıyor (bkz. ADR-003) — sadece paylaşılan
   token'lara bakıyor.
2. `rag/reranker.py`'nin BM25 tokenizasyonu (`text.lower().split()`) Türkçe morfolojiyi
   kök'e indirmiyor — "bloke", "blokla", "bloklamak" BM25 için üç ayrı, ilgisiz token.

Bu, hibrit yaklaşımın *yanlış* olduğu anlamına gelmiyor — gerçek bir embedding modeli
(`EMBEDDING_PROVIDER=openai`) devreye girdiğinde 1. madde ortadan kalkar ve BM25 sadece
tamamlayıcı bir sinyal olarak kalır. Asıl payı çıkan ders: **offline/fake mod, üretim
kalitesinde retrieval iddiasında değil** — zaten README ve ADR-003 bunu söylüyor, bu ölçüm
o iddiayı sayıyla doğruluyor. `tests/unit/test_eval_harness.py`'deki eşik (`>= 0.5`) bu
ölçülen tabanı yansıtıyor; amaç "harika" demek değil, gelecekte bir regresyonu yakalamak.
