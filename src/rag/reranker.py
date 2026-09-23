"""Vektör benzerliğiyle harmanlanmış sözcüksel (BM25) yeniden sıralama.

Sadece vektör benzerliği tam terim eşleşmelerini ("FAST", "KVKK") kaçırabiliyor
— %50/%50 harmanlamak, bir cross-encoder'ın maliyeti olmadan hem anlamsal hem
sözcüksel eşleşmeleri kapsıyor.
"""

from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from nlp.text_utils import ascii_fold
from schemas.dto import Citation

_SNIPPET_LENGTH = 200

# Türkçe sondan eklemeli: "şifre" -> "şifremi", "şikayet" -> "şikayetimi",
# "bloke" -> "bloke ederim". Boşlukla bölen bir BM25 bu çekimleri farklı
# terim sayıyor ve doğru doküman hiç eşleşmiyordu.
#
# Tam bir Türkçe stemmer yerine sabit uzunlukta önek kesme: kök çoğunlukla
# ilk 5 harfte bitiyor ve ek sonra geliyor. Kaba, ama bağımlılık eklemiyor ve
# ölçüldü (bkz. ADR-015): 5 en iyi, 4 ve 6 daha kötü. Aradaki fark iki setin
# birinde birkaç sorgu, yani bu değer "en iyi" değil "ölçülmüş makul" —
# sağlayıcı ya da korpus değişirse yeniden ölçülmeli.
_STEM_LENGTH = 5

# Harmanın varsayılan vektör payı. `Settings.rag_vector_weight` bunu eziyor;
# burada duruyor ki reranker LangGraph ya da ayar nesnesi kurmadan da
# çağrılabilsin.
_DEFAULT_VECTOR_WEIGHT = 0.25


def tokenize(text: str) -> list[str]:
    """BM25 için sözcük listesi: ASCII'ye indirgenmiş, önekten kesilmiş."""
    return [
        word[:_STEM_LENGTH] if len(word) > _STEM_LENGTH else word
        for word in ascii_fold(text).split()
    ]


def _clean_snippet(text: str, limit: int = _SNIPPET_LENGTH) -> str:
    """Alıntı önizlemesini kelime ortasında kesmeyen kırpma.

    Ham `text[:200]` ekranda "Mobil uy" gibi yarım kelimeler bırakıyordu —
    kullanıcıya bozuk görünüyor. Önce cümle sonu aranıyor, yoksa son tam
    kelimede kesilip üç nokta konuyor.
    """
    # Markdown başlığı ("# Hesap İşletim Ücretleri") cevabın başına
    # yapışıyordu: doküman içinde anlamlı ama sohbette "# Hesap İşletim
    # Ücretleri DemoBank A.Ş., hesap paketine göre..." diye tek cümleye
    # dönüşüyor ve okunaksız. Başlık `Citation.title` alanında zaten var.
    lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    collapsed = " ".join(" ".join(lines).split()) or " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    window = collapsed[:limit]
    sentence_end = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if sentence_end >= limit // 2:
        return window[: sentence_end + 1]
    word_end = window.rfind(" ")
    return (window[:word_end] if word_end > 0 else window).rstrip(" ,;:") + "…"


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lowest, highest = min(values), max(values)
    if highest == lowest:
        return [1.0 for _ in values]
    return [(value - lowest) / (highest - lowest) for value in values]


def _to_citation(document: Document, score: float) -> Citation:
    metadata = document.metadata
    return Citation(
        doc_id=str(metadata.get("doc_id", "")),
        title=str(metadata.get("title", "")),
        source=str(metadata.get("source", "")),
        snippet=_clean_snippet(document.page_content),
        score=max(0.0, min(1.0, score)),
    )


def rerank_with_bm25(
    query: str,
    candidates: list[tuple[Document, float]],
    top_k: int = 4,
    vector_weight: float = _DEFAULT_VECTOR_WEIGHT,
) -> list[Citation]:
    if not candidates:
        return []

    # BM25'in IDF'i küçük korpusta dejenere: tek adayda tanımsız, iki adayda
    # tüm terimler idf=0 alıyor ve sözcüksel kanal tamamen susuyor. İkinci
    # durumda `_min_max_normalize` eşit skorları 1.0'a çevirdiği için sıralama
    # zaten vektöre düşüyor, yani davranış doğru — ama sessiz. Aday havuzunun
    # neden dar tutulmaması gerektiğinin bir sebebi daha (bkz. ADR-015).
    if len(candidates) == 1:
        document, vector_score = candidates[0]
        return [_to_citation(document, vector_score)]

    documents = [document for document, _ in candidates]
    vector_scores = [score for _, score in candidates]

    tokenized_corpus = [tokenize(document.page_content) for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = list(bm25.get_scores(tokenize(query)))

    normalized_vector = _min_max_normalize(vector_scores)
    normalized_bm25 = _min_max_normalize(bm25_scores)
    lexical_weight = 1.0 - vector_weight
    combined_scores = [
        vector_weight * vector + lexical_weight * bm25_score
        for vector, bm25_score in zip(normalized_vector, normalized_bm25, strict=True)
    ]

    ranked = sorted(
        zip(documents, combined_scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [_to_citation(document, score) for document, score in ranked[:top_k]]
