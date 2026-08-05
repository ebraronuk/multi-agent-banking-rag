"""Settings yüklenmeden önce tüm test paketini offline/fake sağlayıcılara zorlar.

Bu olmadan, bir testte `app.core.config.get_settings()`'i import etmek
geliştiricinin gerçek `.env`'inde ne varsa onu (ör. LLM_PROVIDER=anthropic)
alırdı, testleri deterministik-olmayan ve ağ + API kredisine bağımlı yapardı.
Testlerin geçmesi için asla gerçek bir anahtar gerekmemeli.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("APP_ENV", "local")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/vectorstore-test")


@pytest.fixture(autouse=True)
def _reset_rate_limiter() -> None:
    """`app.core.rate_limit.limiter`, bu oturumdaki her test tarafından
    paylaşılan modül-seviyesi bir singleton — sıfırlanmazsa, /chat rate
    limitini bilinçli olarak tetikleyen bir test, aynı süreçte/aynı in-memory
    sayaçla çalışan sonraki testleri, gerçekte test ettikleri şeyle hiç ilgisi
    olmayan 429'larla başarısız bırakırdı."""
    from app.core.rate_limit import limiter

    limiter.reset()
