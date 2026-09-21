"""Tracing katmanı testleri.

Buradaki asıl iddia şu: gözlemlenebilirlik hiçbir koşulda isteği düşürmemeli.
Anahtar yoksa, paket kurulu değilse, Langfuse erişilemiyorsa — sistem aynen
çalışmalı. Testlerin çoğu bu yüzden "kapalıyken de doğru davranıyor mu" diye
soruyor, "trace gönderildi mi" diye değil.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.tracing import (
    _should_sample,
    build_trace_config,
    flush_traces,
    get_trace_callbacks,
    tracing_status,
)


def _settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]


class TestTracingEnabled:
    def test_anahtarsiz_kapali(self) -> None:
        assert _settings().tracing_enabled is False

    def test_tek_anahtar_yetmez(self) -> None:
        assert _settings(langfuse_public_key="pk-x").tracing_enabled is False
        assert _settings(langfuse_secret_key="sk-x").tracing_enabled is False

    def test_iki_anahtar_acar(self) -> None:
        settings = _settings(langfuse_public_key="pk-x", langfuse_secret_key="sk-x")
        assert settings.tracing_enabled is True

    def test_enabled_false_anahtarlari_ezer(self) -> None:
        """CI'da anahtarlar ortamda olsa bile tracing kapatılabilmeli."""
        settings = _settings(
            langfuse_public_key="pk-x", langfuse_secret_key="sk-x", langfuse_enabled=False
        )
        assert settings.tracing_enabled is False


class TestCallbacks:
    def test_kapaliyken_bos_liste(self) -> None:
        """`[]` dönmek bilinçli: çağıran taraf `if` yazmak zorunda kalmıyor."""
        assert get_trace_callbacks(_settings(), "conv-1") == []

    def test_sample_rate_sifir_hicbir_sey_gondermez(self) -> None:
        settings = _settings(
            langfuse_public_key="pk-x", langfuse_secret_key="sk-x", langfuse_sample_rate=0.0
        )
        assert get_trace_callbacks(settings, "conv-1") == []


class TestSampling:
    def test_tam_oran_her_zaman_ornekler(self) -> None:
        settings = _settings(langfuse_sample_rate=1.0)
        assert all(_should_sample(settings, f"conv-{i}") for i in range(20))

    def test_sifir_oran_hic_ornekle(self) -> None:
        settings = _settings(langfuse_sample_rate=0.0)
        assert not any(_should_sample(settings, f"conv-{i}") for i in range(20))

    def test_ayni_konusma_ayni_karar(self) -> None:
        """Deterministik olmalı — aynı konuşmanın turn'lerinin yarısını
        kaybeden bir trace okunamaz hale gelir."""
        settings = _settings(langfuse_sample_rate=0.5)
        first = _should_sample(settings, "conv-sabit")
        assert all(_should_sample(settings, "conv-sabit") is first for _ in range(10))


class TestTraceConfig:
    def test_config_langgraph_icin_gecerli(self) -> None:
        config = build_trace_config(_settings(), conversation_id="conv-7")
        assert config["callbacks"] == []
        assert isinstance(config["metadata"], dict)

    def test_konusma_id_session_olarak_gecer(self) -> None:
        """Bir konuşmanın bütün turn'leri Langfuse'da tek oturumda toplanmalı."""
        config = build_trace_config(_settings(), conversation_id="conv-7")
        assert config["metadata"]["langfuse_session_id"] == "conv-7"

    def test_request_id_verilirse_eklenir(self) -> None:
        config = build_trace_config(_settings(), conversation_id="c", request_id="req-9")
        assert config["metadata"]["request_id"] == "req-9"

    def test_request_id_yoksa_anahtar_da_yok(self) -> None:
        config = build_trace_config(_settings(), conversation_id="c")
        assert "request_id" not in config["metadata"]

    def test_varsayilan_etiket_ortam(self) -> None:
        config = build_trace_config(_settings(), conversation_id="c")
        assert config["metadata"]["langfuse_tags"] == ["local"]

    def test_ozel_etiketler_gecer(self) -> None:
        config = build_trace_config(_settings(), conversation_id="c", tags=["eval", "regresyon"])
        assert config["metadata"]["langfuse_tags"] == ["eval", "regresyon"]


class TestStatus:
    def test_durum_anahtar_sizdirmaz(self) -> None:
        settings = _settings(langfuse_public_key="pk-gizli", langfuse_secret_key="sk-gizli")
        serialized = str(tracing_status(settings))
        assert "pk-gizli" not in serialized
        assert "sk-gizli" not in serialized

    def test_kapaliyken_host_gizli(self) -> None:
        assert tracing_status(_settings())["host"] is None


def test_flush_paket_yokken_patlamaz() -> None:
    """Kapanış yolu hiçbir zaman exception fırlatmamalı."""
    flush_traces()


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_gecersiz_sample_rate_reddedilir(rate: float) -> None:
    with pytest.raises(ValueError):
        _settings(langfuse_sample_rate=rate)
