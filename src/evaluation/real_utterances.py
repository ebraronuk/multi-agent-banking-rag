"""Gerçek kullanıcı dili değerlendirmesi.

Neden ayrı bir set: ilk eval setim ("Bakiyem ne kadar acaba?", "Bir müşteri
temsilcisiyle görüşebilir miyim?") tam cümlelerden oluşuyordu ve sistem orada
%100 alıyordu. Sorun şu ki kimse bir chatbot'a öyle yazmıyor.

Gerçek kullanıcı `bakiye` yazar. `param ne kadar kaldı` yazar. `kartimi
kaybettim napcam` yazar — küçük harf, noktalama yok, Türkçe karakter yok.
Temiz bir sette %100 almak, sistemin iyi olduğunu değil, setin kolay olduğunu
gösteriyordu.

Bu modül `data/eval/real_user_utterances.json`'ı okuyup zorluk kategorisi
bazında rapor üretiyor. Kategori kırılımı önemli: toplam skor tek başına
nerede battığını söylemiyor, "Türkçe karaktersiz girdide %40" söylüyor.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from evaluation.report import LatencyStats, MetricResult, _timed
from nlp.intent_classifier import classify_intent_rule_based
from nlp.ner_extractor import extract_entities
from schemas.dto import IntentLabel

DEFAULT_CORPUS_PATH = Path("data/eval/real_user_utterances.json")


@dataclass(frozen=True)
class Utterance:
    text: str
    expected: IntentLabel | None
    difficulty: str
    note: str | None = None


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> tuple[Utterance, ...]:
    """Korpusu okur. Dosya yoksa boş döner — eval, veri dosyasının varlığına
    bağlı olarak çökmemeli."""
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[Utterance] = []
    for row in raw.get("vakalar", []):
        expected_raw = row.get("beklenen_niyet")
        # `null` beklenen niyet = bağlama bağlı ya da tanımsız vaka. Bunlar
        # doğruluk skoruna girmiyor ama çökme testine giriyor: sistem bir
        # cevap üretmek zorunda değil, ama patlamak da zorunda değil.
        expected = IntentLabel(expected_raw) if expected_raw else None
        cases.append(
            Utterance(
                text=row["metin"],
                expected=expected,
                difficulty=row.get("zorluk", "bilinmiyor"),
                note=row.get("not"),
            )
        )
    return tuple(cases)


@dataclass(frozen=True)
class DifficultyBreakdown:
    difficulty: str
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass(frozen=True)
class RealUtteranceResult:
    metric: MetricResult
    breakdown: tuple[DifficultyBreakdown, ...]
    crashes: tuple[str, ...]

    def render(self) -> str:
        lines = [
            "",
            "=" * 62,
            "GERÇEK KULLANICI DİLİ DEĞERLENDİRMESİ",
            "=" * 62,
            f"genel doğruluk       {self.metric.accuracy:>6.1%}  "
            f"({self.metric.correct}/{self.metric.total})   "
            f"p95 {self.metric.latency.p95_ms:>6.1f}ms",
            "",
            "zorluk kırılımı:",
        ]
        for item in sorted(self.breakdown, key=lambda b: b.accuracy):
            bar = "#" * int(item.accuracy * 20)
            lines.append(f"  {item.difficulty:<22} {item.accuracy:>6.1%} ({item.correct}/{item.total})  {bar}")
        if self.crashes:
            lines.append("")
            lines.append("ÇÖKEN GİRDİLER (niyet yanlışlığından daha ciddi):")
            for crash in self.crashes:
                lines.append(f"  - {crash}")
        if self.metric.misses:
            lines.append("")
            lines.append("yanlış sınıflananlar:")
            for miss in self.metric.misses:
                lines.append(f"  - {miss}")
        lines.append("=" * 62)
        return "\n".join(lines)


def run_real_utterance_eval(cases: tuple[Utterance, ...] | None = None) -> RealUtteranceResult:
    """Kural tabanlı niyet yolunu gerçek kullanıcı diline karşı ölçer.

    İki ayrı şey ölçülüyor:
    1. **Doğruluk** — beklenen niyeti olan vakalarda.
    2. **Dayanıklılık** — boş mesaj, tek karakter, anlamsız girdi çökertiyor mu?
       Çökme, yanlış sınıflandırmadan daha ciddi bir hata.
    """
    corpus = cases if cases is not None else load_corpus()
    correct = 0
    scored = 0
    misses: list[str] = []
    crashes: list[str] = []
    samples: list[float] = []
    by_difficulty: dict[str, list[bool]] = defaultdict(list)

    for case in corpus:
        try:
            (predicted, _confidence), elapsed = _timed(
                lambda c=case: classify_intent_rule_based(c.text, extract_entities(c.text))
            )
            samples.append(elapsed)
        except Exception as exc:
            crashes.append(f"{case.text!r} -> {type(exc).__name__}: {exc}")
            continue

        if case.expected is None:
            continue
        scored += 1
        hit = predicted == case.expected
        by_difficulty[case.difficulty].append(hit)
        if hit:
            correct += 1
        else:
            suffix = f"  # {case.note}" if case.note else ""
            misses.append(f"[{case.difficulty}] {case.text!r} beklenen={case.expected} gelen={predicted}{suffix}")

    metric = MetricResult(
        name="real_utterance_accuracy",
        accuracy=correct / scored if scored else 0.0,
        correct=correct,
        total=scored,
        latency=LatencyStats.from_samples(samples),
        misses=tuple(misses),
    )
    breakdown = tuple(
        DifficultyBreakdown(difficulty=name, correct=sum(hits), total=len(hits))
        for name, hits in by_difficulty.items()
    )
    return RealUtteranceResult(metric=metric, breakdown=breakdown, crashes=tuple(crashes))


if __name__ == "__main__":
    print(run_real_utterance_eval().render())
