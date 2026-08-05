"""Force the whole test suite onto the offline/fake providers before Settings loads.

Without this, importing `app.core.config.get_settings()` in a test would pick up
whatever's in a developer's real `.env` (e.g. LLM_PROVIDER=anthropic), making
tests non-deterministic and dependent on network + API credits. Tests should
never need a real key to pass.
"""

from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("APP_ENV", "local")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/vectorstore-test")
