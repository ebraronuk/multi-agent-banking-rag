"""Langfuse tracing — ajan grafiğinin her adımını görünür kılar.

Neden var: LLM sistemlerinde zor olan hatayı görmek değil, **hangi katmanda**
olduğunu görmek. Bir cevap yanlış geldiğinde suçlu genelde modele yazılır; oysa
kaynak çoğu zaman yönlendirme, getirilen bağlam ya da prompt oluyor. Grafik
çağrısının tamamı trace'lendiğinde bu ayrım tahmin olmaktan çıkıp ölçüm haline
geliyor.

Tasarım kararları:

* **Eksik yapılandırma çalışmayı durdurmaz.** Anahtar yoksa `[]` döner ve sistem
  tracing olmadan aynen çalışır. `LLM_PROVIDER=fake` ile aynı felsefe.
* **`langfuse` paketi kurulu olmayabilir.** Import lazy ve korumalı; paket
  yoksa bir kez uyarır, sonra sessizce devre dışı kalır. CI'ı bir gözlem
  aracının varlığına bağlamak istemiyoruz.
* **Sürüm farkı soyutlanmıştır.** Langfuse v3 `langfuse.langchain`, v2
  `langfuse.callback` altından `CallbackHandler` veriyor ve metadata'yı farklı
  aktarıyorlar. Çağıran taraf bunu bilmek zorunda değil.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from app.core.config import Settings
from app.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - yalnızca tip denetimi için
    from langchain_core.callbacks.base import BaseCallbackHandler

logger = get_logger(__name__)

# Langfuse, LangChain run config'i üzerinden bu anahtarlarla metadata okuyor
# (v3). v2'de aynı bilgi handler kurucusuna veriliyor; `build_trace_config`
# ikisini de doldurur.
_SESSION_KEY = "langfuse_session_id"
_USER_KEY = "langfuse_user_id"
_TAGS_KEY = "langfuse_tags"


class _HandlerFactory:
    """Kurulu Langfuse sürümünü bir kez çözer, sonra yeniden kullanır."""

    def __init__(self) -> None:
        self.available: bool = False
        self.version: str = "none"
        self._handler_cls: Any = None
        self._resolve()

    def _resolve(self) -> None:
        try:  # Langfuse v3
            from langfuse.langchain import CallbackHandler

            self._handler_cls = CallbackHandler
            self.version = "v3"
            self.available = True
            return
        except ImportError:
            pass

        try:  # Langfuse v2
            from langfuse.callback import CallbackHandler

            self._handler_cls = CallbackHandler
            self.version = "v2"
            self.available = True
            return
        except ImportError:
            pass

        logger.info(
            "tracing_disabled_package_missing",
            hint="pip install 'langfuse>=2.0' ya da proje ekstrası: pip install -e '.[tracing]'",
        )

    def build(self, settings: Settings) -> Any | None:
        if not self.available or self._handler_cls is None:
            return None
        try:
            if self.version == "v3":
                # v3'te istemci ortam/konstruktör üzerinden ayrı kuruluyor;
                # handler parametresiz alınıyor.
                from langfuse import Langfuse

                Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                return self._handler_cls()
            return self._handler_cls(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except Exception as exc:  # pragma: no cover - ağ/anahtar hatası
            # Gözlemlenebilirlik hiçbir koşulda isteği düşürmemeli.
            logger.warning("tracing_handler_init_failed", error=str(exc))
            return None


@lru_cache(maxsize=1)
def _factory() -> _HandlerFactory:
    return _HandlerFactory()


def tracing_status(settings: Settings) -> dict[str, object]:
    """Sağlık ucu ve testler için okunabilir durum. Anahtar sızdırmaz."""
    factory = _factory()
    return {
        "enabled": settings.tracing_enabled and factory.available,
        "configured": settings.tracing_enabled,
        "package": factory.version,
        "host": settings.langfuse_host if settings.tracing_enabled else None,
        "sample_rate": settings.langfuse_sample_rate,
    }


def _should_sample(settings: Settings, conversation_id: str) -> bool:
    """Deterministik örnekleme.

    Aynı konuşma ya hep trace'lenir ya hiç — rastgele seçim, bir konuşmanın
    turn'lerinin yarısını kaybettirir ve trace'i okunamaz hale getirir.
    """
    rate = settings.langfuse_sample_rate
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    bucket = (hash(conversation_id) % 10_000) / 10_000
    return bucket < rate


def get_trace_callbacks(settings: Settings, conversation_id: str) -> list[BaseCallbackHandler]:
    """Grafik çağrısına verilecek callback listesi. Kapalıysa boş liste.

    Boş liste dönmek bilinçli: çağıran taraf `if` yazmak zorunda kalmıyor,
    `config={"callbacks": []}` LangGraph için geçerli bir girdi.
    """
    if not settings.tracing_enabled:
        return []
    if not _should_sample(settings, conversation_id):
        return []
    handler = _factory().build(settings)
    return [handler] if handler is not None else []


def build_trace_config(
    settings: Settings,
    *,
    conversation_id: str,
    request_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """LangGraph `ainvoke` için hazır config: callbacks + Langfuse metadata.

    `conversation_id` session olarak geçiyor, böylece Langfuse arayüzünde bir
    konuşmanın bütün turn'leri tek bir oturum altında toplanıyor — bir hatayı
    incelerken ihtiyaç duyulan ilk şey o.
    """
    callbacks = get_trace_callbacks(settings, conversation_id)
    metadata: dict[str, Any] = {
        _SESSION_KEY: conversation_id,
        _TAGS_KEY: tags or [settings.app_env.value],
    }
    if request_id:
        metadata["request_id"] = request_id
    return {"callbacks": callbacks, "metadata": metadata}


def flush_traces() -> None:
    """Süreç kapanırken bekleyen trace'leri gönder.

    Langfuse arka planda batch'liyor; kapanışta flush edilmezse son istekler
    kaybolur. Hata yutuluyor: kapanış yolu hiçbir zaman exception fırlatmamalı.
    """
    if not _factory().available:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:  # pragma: no cover
        try:
            from langfuse import Langfuse

            Langfuse().flush()
        except Exception:
            logger.debug("tracing_flush_skipped")
