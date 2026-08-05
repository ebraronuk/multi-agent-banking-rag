"""Tiny, honest evaluation harness — not RAGAS, not a benchmark suite.

Exists to make one point concretely: a multi-agent system's quality claims
should be backed by *some* repeatable measurement, even a small hand-labeled
one, rather than "it seemed to work when I tried it a few times". Swap
`INTENT_EVAL_SET` for a real labeled dataset (ideally sourced from actual
support transcripts) and this becomes a real regression gate that CI can run
on every PR; today it's a worked example of the shape that gate should take.
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
    # Substring expected in the top citation's `source` filename — not an
    # exact match, since chunk filenames carry a `-<index>` suffix
    # (see rag/ingest.py) that would make this brittle for no benefit.
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
    """Does the top citation for each query actually come from the doc that answers it?

    This is retrieval precision@1, not answer-quality — it doesn't touch the
    LLM at all, deliberately: a wrong citation is a retriever bug, a bad
    phrasing of a right citation is a prompt/model concern, and conflating
    the two in one metric makes both harder to debug. RAG answer quality
    itself would need a judged/labeled dataset this project doesn't have;
    swap-in point is the same shape as `run_intent_eval`.
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

    # A throwaway collection in a temp dir, not the app's real
    # `data/vectorstore` — `add_documents` has no dedup-by-content, so
    # re-running this against the live collection would silently pile up
    # duplicate chunks on every invocation instead of giving a clean number.
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
