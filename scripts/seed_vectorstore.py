"""Standalone CLI: seeds the Chroma vector store from `data/sample_docs`.

Run as `python scripts/seed_vectorstore.py` from the repo root (also wired to
`make seed`). Manually prepends `src` to `sys.path` before any project import
because this script runs outside pytest and therefore doesn't get the
`pythonpath = ["src"]` setting from `pyproject.toml`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from rag.ingest import load_sample_documents  # noqa: E402
from rag.vectorstore import build_vectorstore  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    vectorstore = build_vectorstore(settings)

    documents = load_sample_documents()
    if not documents:
        logger.warning("no_sample_documents_found")
        print("No sample documents found under data/sample_docs — nothing ingested.")
        return

    vectorstore.add_documents(documents)
    print(f"Ingested {len(documents)} chunk(s) into the '{settings.chroma_collection}' collection.")


if __name__ == "__main__":
    main()
