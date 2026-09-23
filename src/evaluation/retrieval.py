"""Retrieval top-1 değerlendirmesi ve parametre taraması.

Neden var: hibrit retriever'ın iki ayarı (`rag_candidate_pool`,
`rag_vector_weight`) ve reranker'ın gövde uzunluğu, ölçmeden bakıldığında
tamamen makul görünen sayılar. Ölçünce görünen şey başkaydı — anahtarsız demo
backend'inde vektör kanalı ilgisiz sorgulara ilgili sorgulardan yüksek skor
verebiliyor, yani aday havuzunu dar tutmak doğru dokümanı sessizce eliyordu
(ayrıntı: ADR-015).

Set ikiye bölünmüş durumda ve bu bölme bu modülün asıl sebebi:

* `ayar`      — parametreler bu sete bakılarak seçildi.
* `dogrulama` — seçerken hiç kullanılmadı.

İlk taramada `ayar` setinde %66.7'den %88.9'a çıkan bir yapılandırma,
`dogrulama` setinde hiçbir şey kazandırmadı: kazanç gerçek değil, sete
uydurmaydı. Tek bir sette ölçülen iyileşme, iyileşme sayılmıyor.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from app.core.config import Settings, get_settings
from rag.ingest import load_sample_documents
from rag.retriever import HybridRetriever
from rag.vectorstore import build_vectorstore

DEFAULT_QUERIES_PATH = Path("data/eval/retrieval_queries.json")


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    expected_source: str
    split: str


@dataclass(frozen=True)
class SplitResult:
    split: str
    total: int
    correct: int
    failures: tuple[tuple[str, str, str], ...]  # (sorgu, beklenen, gelen)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def load_cases(path: Path = DEFAULT_QUERIES_PATH) -> tuple[RetrievalCase, ...]:
    """Sorguları okur. Dosya yoksa boş döner — eval, veri dosyasının varlığına
    bağlı olarak çökmemeli (bkz. `real_utterances.load_corpus`)."""
    if not path.exists():
        return ()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        RetrievalCase(
            query=row["metin"],
            expected_source=row["beklenen_kaynak"],
            split=row.get("bolum", "ayar"),
        )
        for row in raw.get("sorgular", [])
    )


def evaluate(retriever: HybridRetriever, cases: tuple[RetrievalCase, ...]) -> list[SplitResult]:
    """Her bölüm için top-1 doğruluğu. Sıralama giriş sırasını koruyor ki
    rapor her koşuda aynı düzende okunsun."""
    by_split: dict[str, list[RetrievalCase]] = {}
    for case in cases:
        by_split.setdefault(case.split, []).append(case)

    results: list[SplitResult] = []
    for split, split_cases in by_split.items():
        failures: list[tuple[str, str, str]] = []
        correct = 0
        for case in split_cases:
            citations = retriever.retrieve(case.query)
            got = citations[0].source if citations else "(sonuç yok)"
            if got == case.expected_source:
                correct += 1
            else:
                failures.append((case.query, case.expected_source, got))
        results.append(
            SplitResult(
                split=split, total=len(split_cases), correct=correct, failures=tuple(failures)
            )
        )
    return results


def render(results: list[SplitResult]) -> str:
    lines = ["Retrieval top-1 doğruluğu", "=" * 40]
    for result in results:
        lines.append(f"{result.split:12s} {result.correct:>2d}/{result.total:<2d}  {result.accuracy:.1%}")
    total = sum(r.total for r in results)
    correct = sum(r.correct for r in results)
    if total:
        lines.append(f"{'toplam':12s} {correct:>2d}/{total:<2d}  {correct / total:.1%}")
    for result in results:
        if result.failures:
            lines.append(f"\n{result.split} bölümünde ıskalananlar:")
            lines.extend(
                f"  {query}\n      beklenen={expected}  gelen={got}"
                for query, expected, got in result.failures
            )
    return "\n".join(lines)


@dataclass(frozen=True)
class SweepRow:
    pool: int
    vector_weight: float
    accuracy_by_split: dict[str, float]


def sweep(
    settings: Settings,
    cases: tuple[RetrievalCase, ...],
    pools: tuple[int, ...] = (8, 12, 16, 24, 34),
    weights: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0),
) -> Iterator[SweepRow]:
    """Parametre ızgarası. Her satır iki bölümü de ayrı ayrı raporluyor —
    tek bir birleşik sayı, uydurmayı gizler."""
    with _isolated_vectorstore(settings) as eval_settings:
        store = build_vectorstore(eval_settings)
        for pool in pools:
            for weight in weights:
                retriever = HybridRetriever(store, k_vector=pool, vector_weight=weight)
                results = evaluate(retriever, cases)
                yield SweepRow(
                    pool=pool,
                    vector_weight=weight,
                    accuracy_by_split={r.split: r.accuracy for r in results},
                )


class _isolated_vectorstore:  # noqa: N801 — context manager, sınıf adı değil kullanımı önemli
    """Geçici, tek seferlik koleksiyon.

    `add_documents` dedup yapmıyor; canlı koleksiyona karşı ölçmek her koşuda
    chunk biriktirip skoru kaydırıyor (aynı tuzak `evaluation/__main__.py`'de
    de var).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tmp = tempfile.TemporaryDirectory()

    def __enter__(self) -> Settings:
        eval_settings = self._settings.model_copy(
            update={"chroma_persist_dir": self._tmp.name, "chroma_collection": "eval-retrieval"}
        )
        build_vectorstore(eval_settings).add_documents(load_sample_documents())
        return eval_settings

    def __exit__(self, *exc: object) -> None:
        self._tmp.cleanup()


def run(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="evaluation.retrieval")
    parser.add_argument("--sweep", action="store_true", help="parametre ızgarasını tara")
    args = parser.parse_args(argv)

    cases = load_cases()
    if not cases:
        print(f"Sorgu dosyası bulunamadı: {DEFAULT_QUERIES_PATH}")
        return 0

    settings = get_settings()
    if args.sweep:
        print(f"{'havuz':>6s} {'vektör payı':>12s}   bölüm bazlı top-1")
        for row in sweep(settings, cases):
            detail = "  ".join(f"{name}={value:.1%}" for name, value in row.accuracy_by_split.items())
            print(f"{row.pool:>6d} {row.vector_weight:>12.2f}   {detail}")
        return 0

    with _isolated_vectorstore(settings) as eval_settings:
        retriever = HybridRetriever(
            build_vectorstore(eval_settings),
            k_vector=eval_settings.rag_candidate_pool,
            vector_weight=eval_settings.rag_vector_weight,
        )
        print(render(evaluate(retriever, cases)))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
