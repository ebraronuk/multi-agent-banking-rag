"""Force the whole test suite onto the offline/fake providers before Settings loads.

Without this, importing `app.core.config.get_settings()` in a test would pick up
whatever's in a developer's real `.env` (e.g. LLM_PROVIDER=anthropic), making
tests non-deterministic and dependent on network + API credits. Tests should
never need a real key to pass.
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
    """`app.core.rate_limit.limiter` is a module-level singleton shared by
    every test in this session — without resetting it, a test that
    intentionally trips the /chat rate limit would leave later tests
    (running in the same process, same in-memory counter) failing with 429s
    that have nothing to do with what they're actually testing."""
    from app.core.rate_limit import limiter

    limiter.reset()
