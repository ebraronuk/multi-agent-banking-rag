"""Retrieval değerlendirme harness'ı.

Buradaki testler doğruluğun belirli bir sayının üstünde olmasını İDDİA ETMİYOR
— o sayı korpusa ve embedding sağlayıcısına bağlı ve `python -m
evaluation.retrieval` ile ölçülüyor. Test edilen şey harness'ın kendisi: bölüm
ayrımını koruyor mu, veri dosyası yokken çöküyor mu, ıskalananları doğru
raporluyor mu.

Ayrımın kendisi de test ediliyor (`test_iki_bolum_de_var`) çünkü onu kaybetmek
tüm değerlendirmeyi işe yaramaz hale getirir: tek bir sette ölçülen iyileşme,
o sete uydurmadan ayırt edilemez.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.retrieval import (
    DEFAULT_QUERIES_PATH,
    RetrievalCase,
    SplitResult,
    evaluate,
    load_cases,
    render,
)
from schemas.dto import Citation


class _StubRetriever:
    """Sabit cevap veren retriever — harness'ı vektör deposu kurmadan test eder."""

    def __init__(self, answer: str) -> None:
        self.answer = answer

    def retrieve(self, query: str) -> list[Citation]:
        return [Citation(doc_id="d", title="t", source=self.answer, snippet="s", score=1.0)]


def test_veri_dosyasi_okunuyor() -> None:
    cases = load_cases()
    assert cases, f"{DEFAULT_QUERIES_PATH} boş ya da bulunamadı"


def test_iki_bolum_de_var() -> None:
    """Ayrım kaybolursa değerlendirme kendini kandırmaya geri döner."""
    splits = {case.split for case in load_cases()}
    assert {"ayar", "dogrulama"} <= splits


def test_beklenen_kaynaklar_gercek_dosyalar() -> None:
    """Yazım hatası bir sorguyu sessizce çözülemez hale getirir."""
    available = {path.name for path in Path("data/sample_docs").glob("*.md")}
    missing = {case.expected_source for case in load_cases()} - available
    assert not missing, f"bilgi tabanında olmayan kaynak: {missing}"


def test_dosya_yoksa_bos_donuyor(tmp_path: Path) -> None:
    assert load_cases(tmp_path / "yok.json") == ()


def test_dogru_cevap_sayiliyor() -> None:
    cases = (RetrievalCase("q", "a.md", "ayar"),)
    (result,) = evaluate(_StubRetriever("a.md"), cases)  # type: ignore[arg-type]
    assert result.correct == 1 and result.failures == ()


def test_yanlis_cevap_iskalanan_olarak_raporlaniyor() -> None:
    cases = (RetrievalCase("q", "a.md", "ayar"),)
    (result,) = evaluate(_StubRetriever("b.md"), cases)  # type: ignore[arg-type]
    assert result.correct == 0
    assert result.failures == (("q", "a.md", "b.md"),)


def test_bolumler_ayri_raporlaniyor() -> None:
    cases = (
        RetrievalCase("q1", "a.md", "ayar"),
        RetrievalCase("q2", "b.md", "dogrulama"),
    )
    results = {r.split: r for r in evaluate(_StubRetriever("a.md"), cases)}  # type: ignore[arg-type]
    assert results["ayar"].accuracy == 1.0
    assert results["dogrulama"].accuracy == 0.0


def test_bos_bolumde_sifira_bolme_yok() -> None:
    assert SplitResult(split="ayar", total=0, correct=0, failures=()).accuracy == 0.0


def test_rapor_her_iki_bolumu_de_yaziyor() -> None:
    cases = (
        RetrievalCase("q1", "a.md", "ayar"),
        RetrievalCase("q2", "b.md", "dogrulama"),
    )
    output = render(evaluate(_StubRetriever("a.md"), cases))  # type: ignore[arg-type]
    assert "ayar" in output and "dogrulama" in output and "toplam" in output
