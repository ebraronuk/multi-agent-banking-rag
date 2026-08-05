"""Ingestion of the sample banking-FAQ knowledge base into chunked Documents.

Chunking (rather than embedding whole files) keeps retrieved context short
and topically focused: a 300-word FAQ doc already covers one policy, but
splitting it still bounds worst-case snippet length and lets the retriever
surface the single paragraph that answers the question instead of the whole
document.
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.logging import get_logger

logger = get_logger(__name__)

_H1_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _extract_title(markdown_text: str, fallback: str) -> str:
    match = _H1_PATTERN.search(markdown_text)
    return match.group(1).strip() if match else fallback


def load_sample_documents(dir_path: str = "data/sample_docs") -> list[Document]:
    base_dir = Path(dir_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)

    documents: list[Document] = []
    file_count = 0

    for file_path in sorted(base_dir.glob("*.md")):
        file_count += 1
        text = file_path.read_text(encoding="utf-8")
        title = _extract_title(text, fallback=file_path.stem)

        for chunk_index, chunk in enumerate(splitter.split_text(text)):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": f"{file_path.name}-{chunk_index}",
                        "title": title,
                        "source": file_path.name,
                    },
                )
            )

    logger.info("sample_documents_loaded", file_count=file_count, chunk_count=len(documents))
    return documents
