"""Küçük, dürüst bir değerlendirme aracı — RAGAS değil, bir benchmark paketi değil.

Tek bir noktayı somut olarak göstermek için var: çoklu-ajan bir sistemin
kalite iddiaları, "birkaç kere denedim, çalışıyor gibiydi" yerine *bir*
tekrarlanabilir ölçümle desteklenmeli, küçük ve elle etiketlenmiş olsa bile.
`INTENT_EVAL_SET`'i gerçek bir etiketli veri setiyle (ideal olarak gerçek
destek transkriptlerinden) değiştirmek, bunu CI'ın her PR'da çalıştırabileceği
gerçek bir regresyon kapısına dönüştürür; bugün bu kapının nasıl bir şekil
alması gerektiğinin somut bir örneği.
"""

from __future__ import annotations

from dataclasses import dataclass

from nlp.intent_classifier import classify_intent_rule_based
from nlp.ner_extractor import extract_entities
from rag.retriever import HybridRetriever
from schemas.dto import IntentLabel


@dataclass(frozen=True)
class IntentEvalCase:
    text: str
    expected: IntentLabel


INTENT_EVAL_SET: tuple[IntentEvalCase, ...] = (
    IntentEvalCase("Merhaba, iyi günler", IntentLabel.SMALL_TALK),
    IntentEvalCase("Bakiyem ne kadar acaba?", IntentLabel.ACCOUNT_ACTION),
    IntentEvalCase("EFT limitiniz ne kadar?", IntentLabel.RAG_QUERY),
    IntentEvalCase("Kartımı çaldılar, bloke edin lütfen", IntentLabel.CARD_ACTION),
    IntentEvalCase("Son işlemlerimi görmek istiyorum", IntentLabel.TRANSACTION_ACTION),
    IntentEvalCase("Bir müşteri temsilcisiyle görüşebilir miyim?", IntentLabel.ESCALATE),
    IntentEvalCase("Yarın hava nasıl olacak?", IntentLabel.OUT_OF_SCOPE),
    IntentEvalCase("Hesap işletim ücretiniz nedir?", IntentLabel.RAG_QUERY),
)


@dataclass(frozen=True)
class IntentEvalResult:
    total: int
    correct: int
    misses: tuple[tuple[str, IntentLabel, IntentLabel], ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_intent_eval(cases: tuple[IntentEvalCase, ...] = INTENT_EVAL_SET) -> IntentEvalResult:
    correct = 0
    misses: list[tuple[str, IntentLabel, IntentLabel]] = []
    for case in cases:
        entities = extract_entities(case.text)
        predicted, _confidence = classify_intent_rule_based(case.text, entities)
        if predicted == case.expected:
            correct += 1
        else:
            misses.append((case.text, case.expected, predicted))
    return IntentEvalResult(total=len(cases), correct=correct, misses=tuple(misses))


@dataclass(frozen=True)
class RetrievalEvalCase:
    query: str
    # En üstteki alıntının `source` dosya adında beklenen alt-dize — birebir
    # eşleşme değil, çünkü chunk dosya adları bir `-<index>` soneki taşıyor
    # (bkz. rag/ingest.py) ve bu, hiçbir kazanç olmadan kırılganlık eklerdi.
    expected_source: str


RETRIEVAL_EVAL_SET: tuple[RetrievalEvalCase, ...] = (
    RetrievalEvalCase("EFT limitiniz ne kadar?", "havale-eft-limitleri"),
    RetrievalEvalCase("Kartımı ne zaman bloke edebilirim?", "kart-engelleme-politikasi"),
    RetrievalEvalCase("Hesap işletim ücreti var mı?", "hesap-isletim-ucretleri"),
    RetrievalEvalCase("Çalışma saatleriniz nedir?", "calisma-saatleri"),
    RetrievalEvalCase("Kişisel verilerim KVKK kapsamında nasıl korunuyor?", "kvkk-gizlilik-notu"),
    RetrievalEvalCase("Şifremi kimseyle paylaşmalı mıyım?", "sifre-guvenlik-onerileri"),
)


@dataclass(frozen=True)
class RetrievalEvalResult:
    total: int
    correct: int
    misses: tuple[tuple[str, str, str], ...]

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_retrieval_eval(
    retriever: HybridRetriever, cases: tuple[RetrievalEvalCase, ...] = RETRIEVAL_EVAL_SET
) -> RetrievalEvalResult:
    """Her sorgu için en üstteki alıntı, gerçekten onu yanıtlayan dokümandan mı geliyor?

    Bu retrieval precision@1, cevap kalitesi değil — bilinçli olarak LLM'e hiç
    dokunmuyor: yanlış bir alıntı bir retriever hatası, doğru bir alıntının
    kötü ifade edilmesi bir prompt/model meselesi — ikisini tek bir metrikte
    karıştırmak ikisini de debug etmeyi zorlaştırırdı. RAG cevap kalitesinin
    kendisi bu projenin sahip olmadığı, hakemli/etiketlenmiş bir veri seti
    gerektirirdi; yerine koyma noktası `run_intent_eval` ile aynı şekilde.
    """
    correct = 0
    misses: list[tuple[str, str, str]] = []
    for case in cases:
        citations = retriever.retrieve(case.query)
        top_source = citations[0].source if citations else ""
        if case.expected_source in top_source:
            correct += 1
        else:
            misses.append((case.query, case.expected_source, top_source or "<no citations>"))
    return RetrievalEvalResult(total=len(cases), correct=correct, misses=tuple(misses))


if __name__ == "__main__":
    import tempfile

    from app.core.config import get_settings
    from rag.ingest import load_sample_documents
    from rag.retriever import build_retriever
    from rag.vectorstore import build_vectorstore

    intent_result = run_intent_eval()
    print(f"intent accuracy: {intent_result.correct}/{intent_result.total} ({intent_result.accuracy:.0%})")
    for text, expected, predicted in intent_result.misses:
        print(f"  MISS: {text!r} expected={expected} got={predicted}")

    # Geçici bir dizinde tek kullanımlık bir koleksiyon, uygulamanın gerçek
    # `data/vectorstore`'u değil — `add_documents`'ın içerik-bazlı bir dedup'ı
    # yok, yani bunu canlı koleksiyona karşı tekrar çalıştırmak temiz bir sayı
    # vermek yerine her çağrıda sessizce tekrarlanan chunk'lar biriktirirdi.
    with tempfile.TemporaryDirectory() as tmp_dir:
        eval_settings = get_settings().model_copy(
            update={"chroma_persist_dir": tmp_dir, "chroma_collection": "eval-harness-cli"}
        )
        vectorstore = build_vectorstore(eval_settings)
        vectorstore.add_documents(load_sample_documents())
        retrieval_result = run_retrieval_eval(build_retriever(eval_settings))

    print(
        f"retrieval precision@1: {retrieval_result.correct}/{retrieval_result.total} "
        f"({retrieval_result.accuracy:.0%})"
    )
    for query, expected_source, got_source in retrieval_result.misses:
        print(f"  MISS: {query!r} expected_source={expected_source!r} got={got_source!r}")
