"""Örnek bankacılık SSS bilgi tabanını parçalanmış Document'lara aktarır.

Tüm dosyayı embed etmek yerine parçalamak (chunking), getirilen bağlamı kısa
ve konuya odaklı tutuyor: 300 kelimelik bir SSS dokümanı zaten tek bir
politikayı kapsıyor, ama yine de parçalamak en kötü durum snippet uzunluğunu
sınırlıyor ve retriever'ın tüm doküman yerine soruyu yanıtlayan tek bir
paragrafı öne çıkarmasını sağlıyor.
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
