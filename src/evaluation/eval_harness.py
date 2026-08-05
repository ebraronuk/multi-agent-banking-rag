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


if __name__ == "__main__":
    result = run_intent_eval()
    print(f"intent accuracy: {result.correct}/{result.total} ({result.accuracy:.0%})")
    for text, expected, predicted in result.misses:
        print(f"  MISS: {text!r} expected={expected} got={predicted}")
