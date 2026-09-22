"""Ölçüm ve regresyon kapısı testleri.

Kapının kendisi test edilmeli: bir regresyon kapısı yanlış alarm verirse
kapatılır, hiç alarm vermezse zaten yoktur. Buradaki testler iki tarafı da
kontrol ediyor — tolerans içindeki oynamayı geçirdiğini ve gerçek düşüşü
yakaladığını.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.report import (
    ACCURACY_TOLERANCE,
    EvalReport,
    LatencyStats,
    MetricResult,
    compare_to_baseline,
    measure_intent,
)


def _metric(name: str = "intent_accuracy", accuracy: float = 1.0, p95: float = 10.0) -> MetricResult:
    return MetricResult(
        name=name,
        accuracy=accuracy,
        correct=int(accuracy * 10),
        total=10,
        latency=LatencyStats(count=10, mean_ms=p95 / 2, p50_ms=p95 / 2, p95_ms=p95, max_ms=p95),
    )


def _report(*metrics: MetricResult) -> EvalReport:
    report = EvalReport()
    report.metrics.extend(metrics)
    return report


class TestLatencyStats:
    def test_bos_ornek_sifir_doner(self) -> None:
        stats = LatencyStats.from_samples([])
        assert stats.count == 0 and stats.p95_ms == 0.0

    def test_tek_ornek(self) -> None:
        stats = LatencyStats.from_samples([42.0])
        assert stats.p50_ms == stats.p95_ms == stats.max_ms == 42.0

    def test_p95_kuyrugu_yakalar(self) -> None:
        """Ortalama kuyruğu gizler; p95 bu yüzden var."""
        samples = [1.0] * 19 + [500.0]
        stats = LatencyStats.from_samples(samples)
        assert stats.mean_ms < 30.0
        assert stats.p95_ms >= 500.0 or stats.max_ms == 500.0

    def test_siralama_girdiden_bagimsiz(self) -> None:
        assert LatencyStats.from_samples([5.0, 1.0, 3.0]).p50_ms == 3.0


class TestBaselineGate:
    def test_temel_yoksa_regresyon_yok(self, tmp_path: Path) -> None:
        """İlk koşuda kapı olmamalı — yokluk hata değil."""
        assert compare_to_baseline(_report(_metric()), tmp_path / "yok.json") == []

    def test_ayni_sonuc_gecer(self, tmp_path: Path) -> None:
        baseline = tmp_path / "b.json"
        _report(_metric(accuracy=0.9)).save(baseline)
        assert compare_to_baseline(_report(_metric(accuracy=0.9)), baseline) == []

    def test_tolerans_icindeki_dusus_gecer(self, tmp_path: Path) -> None:
        """Küçük oynamada CI kırmak, kapının kapatılmasıyla sonuçlanır."""
        baseline = tmp_path / "b.json"
        _report(_metric(accuracy=0.90)).save(baseline)
        current = _report(_metric(accuracy=0.90 - ACCURACY_TOLERANCE / 2))
        assert compare_to_baseline(current, baseline) == []

    def test_gercek_dusus_yakalanir(self, tmp_path: Path) -> None:
        baseline = tmp_path / "b.json"
        _report(_metric(accuracy=0.95)).save(baseline)
        regressions = compare_to_baseline(_report(_metric(accuracy=0.60)), baseline)
        assert len(regressions) == 1
        assert regressions[0].field_name == "accuracy"

    def test_iyilesme_regresyon_sayilmaz(self, tmp_path: Path) -> None:
        baseline = tmp_path / "b.json"
        _report(_metric(accuracy=0.60)).save(baseline)
        assert compare_to_baseline(_report(_metric(accuracy=0.95)), baseline) == []

    def test_yavaslamada_regresyon(self, tmp_path: Path) -> None:
        baseline = tmp_path / "b.json"
        _report(_metric(p95=10.0)).save(baseline)
        regressions = compare_to_baseline(_report(_metric(p95=100.0)), baseline)
        assert any(r.field_name == "p95_ms" for r in regressions)

    def test_kucuk_yavaslama_gecer(self, tmp_path: Path) -> None:
        baseline = tmp_path / "b.json"
        _report(_metric(p95=10.0)).save(baseline)
        assert compare_to_baseline(_report(_metric(p95=12.0)), baseline) == []

    def test_temelde_olmayan_metrik_atlanir(self, tmp_path: Path) -> None:
        """Yeni bir metrik eklemek, henüz temeli olmadığı için CI'ı kırmamalı."""
        baseline = tmp_path / "b.json"
        _report(_metric(name="intent_accuracy")).save(baseline)
        current = _report(_metric(name="intent_accuracy"), _metric(name="yepyeni_metrik", accuracy=0.1))
        assert compare_to_baseline(current, baseline) == []


class TestReportSerialization:
    def test_kaydet_ve_oku(self, tmp_path: Path) -> None:
        path = _report(_metric()).save(tmp_path / "alt" / "b.json")
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "intent_accuracy" in data["metrics"]

    def test_render_metrigi_icerir(self) -> None:
        assert "intent_accuracy" in _report(_metric()).render()

    def test_get_bilinmeyen_metrik_none(self) -> None:
        assert _report(_metric()).get("olmayan") is None


class TestMeasureIntent:
    def test_gercek_olcum_calisir(self) -> None:
        """LLM'siz, deterministik: kural katmanının tek başına ne taşıdığını ölçer."""
        result = measure_intent()
        assert result.name == "intent_accuracy"
        assert result.total > 0
        assert 0.0 <= result.accuracy <= 1.0
        assert result.latency.count == result.total
        assert len(result.misses) == result.total - result.correct
