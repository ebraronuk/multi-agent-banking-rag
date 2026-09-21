"""Ölçüm ve regresyon kapısı katmanı.

`eval_harness` "doğru mu" sorusunu yanıtlıyor. Bu modül iki soru daha ekliyor:
**ne kadar sürdü** ve **dünden kötüye gitti mi**.

İkincisi asıl mesele. Tek seferlik bir doğruluk sayısı iyi hissettirir ama
hiçbir şeyi korumaz; bir prompt değişikliği retrieval'ı bozduğunda bunu
kullanıcıdan önce yakalayan şey, eşiklere bağlanmış ve CI'da çalışan bir
karşılaştırmadır. Bu yüzden rapor JSON'a yazılıyor ve bir sonraki koşu
`--gate` ile ona karşı ölçülüyor.

Bilinçli olarak dışarıda bırakılanlar: RAGAS ve LLM-as-judge. İkisi de
değerlendirmenin kendisini non-deterministik yapar ve regresyon kapısını
gürültüye boğar. Buradaki her metrik deterministik.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from evaluation.eval_harness import (
    INTENT_EVAL_SET,
    RETRIEVAL_EVAL_SET,
    IntentEvalCase,
    RetrievalEvalCase,
)
from nlp.intent_classifier import classify_intent_rule_based
from nlp.ner_extractor import extract_entities
from rag.retriever import HybridRetriever

T = TypeVar("T")

DEFAULT_BASELINE_PATH = Path("data/eval_baseline.json")

# Doğruluk bu kadar puandan fazla düşerse regresyon sayılır. Sıfır değil:
# retrieval skorları embedding sağlayıcısına göre küçük oynamalar gösterir ve
# her oynamada CI'ı kırmak, kapının kapatılmasıyla sonuçlanır.
ACCURACY_TOLERANCE = 0.05
# Gecikme bu oranın üstünde artarsa regresyon. 1.5 = %50 yavaşlama.
LATENCY_TOLERANCE_FACTOR = 1.5


@dataclass(frozen=True)
class LatencyStats:
    """Milisaniye cinsinden. p95 var çünkü ortalama, kuyruğu gizler."""

    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> LatencyStats:
        if not samples:
            return cls(count=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, max_ms=0.0)
        ordered = sorted(samples)
        # Küçük setlerde quantiles() patlar; indeksle almak hem güvenli hem
        # tekrarlanabilir.
        p95_index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return cls(
            count=len(ordered),
            mean_ms=round(statistics.fmean(ordered), 2),
            p50_ms=round(statistics.median(ordered), 2),
            p95_ms=round(ordered[p95_index], 2),
            max_ms=round(ordered[-1], 2),
        )


@dataclass(frozen=True)
class MetricResult:
    name: str
    accuracy: float
    correct: int
    total: int
    latency: LatencyStats
    misses: tuple[str, ...] = ()


@dataclass
class EvalReport:
    metrics: list[MetricResult] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    def get(self, name: str) -> MetricResult | None:
        return next((m for m in self.metrics if m.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "metrics": {m.name: asdict(m) for m in self.metrics},
        }

    def save(self, path: Path = DEFAULT_BASELINE_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def render(self) -> str:
        lines = ["", "=" * 62, "DEĞERLENDİRME RAPORU", "=" * 62]
        for m in self.metrics:
            lines.append(
                f"{m.name:<22} {m.accuracy:>6.1%}  ({m.correct}/{m.total})   "
                f"p50 {m.latency.p50_ms:>7.1f}ms   p95 {m.latency.p95_ms:>7.1f}ms"
            )
            for miss in m.misses:
                lines.append(f"    MISS: {miss}")
        lines.append("=" * 62)
        return "\n".join(lines)


def _timed(fn: Callable[[], T]) -> tuple[T, float]:
    """Çağrıyı çalıştırır, milisaniye cinsinden süresiyle birlikte döner."""
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def measure_intent(cases: tuple[IntentEvalCase, ...] = INTENT_EVAL_SET) -> MetricResult:
    """Niyet sınıflandırması — kural tabanlı yol, LLM'e dokunmadan.

    LLM'siz ölçmek bilinçli: kural katmanının ne kadarını tek başına
    taşıdığını bilmek, model çağrısını atlayan hızlı yolun gerçekten işe
    yarayıp yaramadığını söyleyen tek şey.
    """
    correct = 0
    misses: list[str] = []
    samples: list[float] = []
    for case in cases:
        (predicted, _confidence), elapsed = _timed(
            lambda c=case: classify_intent_rule_based(c.text, extract_entities(c.text))
        )
        samples.append(elapsed)
        if predicted == case.expected:
            correct += 1
        else:
            misses.append(f"{case.text!r} beklenen={case.expected} gelen={predicted}")
    return MetricResult(
        name="intent_accuracy",
        accuracy=correct / len(cases) if cases else 0.0,
        correct=correct,
        total=len(cases),
        latency=LatencyStats.from_samples(samples),
        misses=tuple(misses),
    )


def measure_retrieval(
    retriever: HybridRetriever, cases: tuple[RetrievalEvalCase, ...] = RETRIEVAL_EVAL_SET
) -> MetricResult:
    """Retrieval precision@1 — üstteki alıntı gerçekten soruyu yanıtlayan doküman mı?

    Bu ayrımın maliyeti şurada görülüyor: cevap yanlış geldiğinde suçlu
    genelde modele yazılır. Bu metrik düşükken model değiştirmek zaman kaybı,
    çünkü sorun modele ulaşan bağlamda.
    """
    correct = 0
    misses: list[str] = []
    samples: list[float] = []
    for case in cases:
        citations, elapsed = _timed(lambda c=case: retriever.retrieve(c.query))
        samples.append(elapsed)
        top_source = citations[0].source if citations else ""
        if case.expected_source in top_source:
            correct += 1
        else:
            misses.append(f"{case.query!r} beklenen={case.expected_source!r} gelen={top_source or '<alıntı yok>'!r}")
    return MetricResult(
        name="retrieval_p_at_1",
        accuracy=correct / len(cases) if cases else 0.0,
        correct=correct,
        total=len(cases),
        latency=LatencyStats.from_samples(samples),
        misses=tuple(misses),
    )


def run_all(retriever: HybridRetriever | None = None) -> EvalReport:
    """Tüm deterministik metrikler. Retriever verilmezse yalnız intent ölçülür."""
    report = EvalReport()
    report.metrics.append(measure_intent())
    if retriever is not None:
        report.metrics.append(measure_retrieval(retriever))
    return report


@dataclass(frozen=True)
class Regression:
    metric: str
    field_name: str
    baseline: float
    current: float

    def __str__(self) -> str:
        return (
            f"{self.metric}.{self.field_name}: temel {self.baseline:.4g} -> "
            f"şimdi {self.current:.4g}"
        )


def compare_to_baseline(report: EvalReport, baseline_path: Path = DEFAULT_BASELINE_PATH) -> list[Regression]:
    """Raporu kayıtlı temele karşı ölçer. Temel yoksa boş liste döner.

    Temelin yokluğu hata değil: ilk koşuda kapı yoktur, `--save` ile kurulur.
    """
    if not baseline_path.exists():
        return []
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")).get("metrics", {})
    regressions: list[Regression] = []
    for metric in report.metrics:
        prior = baseline.get(metric.name)
        if not prior:
            continue
        prior_accuracy = float(prior.get("accuracy", 0.0))
        if metric.accuracy < prior_accuracy - ACCURACY_TOLERANCE:
            regressions.append(Regression(metric.name, "accuracy", prior_accuracy, metric.accuracy))
        prior_p95 = float(prior.get("latency", {}).get("p95_ms", 0.0))
        if prior_p95 > 0 and metric.latency.p95_ms > prior_p95 * LATENCY_TOLERANCE_FACTOR:
            regressions.append(Regression(metric.name, "p95_ms", prior_p95, metric.latency.p95_ms))
    return regressions


def push_to_langfuse(report: EvalReport, run_name: str | None = None) -> bool:
    """Metrikleri Langfuse'a skor olarak gönderir. Başarısızlık sessiz.

    Değerlendirme, gözlemlenebilirlik aracının ayakta olmasına bağlı olmamalı:
    `make eval` internet olmadan da çalışmalı.
    """
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.tracing_enabled:
        return False
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        name = run_name or f"eval-{int(report.generated_at)}"
        for metric in report.metrics:
            client.create_score(name=f"{name}.{metric.name}", value=metric.accuracy)
            client.create_score(name=f"{name}.{metric.name}.p95_ms", value=metric.latency.p95_ms)
        client.flush()
        return True
    except Exception:  # pragma: no cover - ağ/sürüm farkı
        return False
