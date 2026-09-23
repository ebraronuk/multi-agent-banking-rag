"""Vektör benzerliğiyle harmanlanmış sözcüksel (BM25) yeniden sıralama.

Sadece vektör benzerliği tam terim eşleşmelerini ("FAST", "KVKK") kaçırabiliyor
— %50/%50 harmanlamak, bir cross-encoder'ın maliyeti olmadan hem anlamsal hem
sözcüksel eşleşmeleri kapsıyor.
"""

from __future__ import annotations

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from nlp.text_utils import turkish_lower
from schemas.dto import Citation

_SNIPPET_LENGTH = 200


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
) -> list[Citation]:
    if not candidates:
        return []

    # BM25'in IDF'i tek dokümanda tanımsız/dejenere — vektör skoruna aynen düş.
    if len(candidates) == 1:
        document, vector_score = candidates[0]
        return [_to_citation(document, vector_score)]

    documents = [document for document, _ in candidates]
    vector_scores = [score for _, score in candidates]

    tokenized_corpus = [turkish_lower(document.page_content).split() for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = list(bm25.get_scores(turkish_lower(query).split()))

    normalized_vector = _min_max_normalize(vector_scores)
    normalized_bm25 = _min_max_normalize(bm25_scores)
    combined_scores = [
        0.5 * vector + 0.5 * bm25_score
        for vector, bm25_score in zip(normalized_vector, normalized_bm25, strict=True)
    ]

    ranked = sorted(
        zip(documents, combined_scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [_to_citation(document, score) for document, score in ranked[:top_k]]
