"""Örnek doküman ingestion/chunking'i için birim testler.

Gerçek `data/sample_docs` korpusu yerine küçük markdown fixture'larının
bulunduğu geçici bir dizin kullanıyor, ki test gerçek bilgi tabanı içeriğine
bağımlı olmasın (biri onu düzenlediğinde bozulmasın).
"""

from __future__ import annotations

from pathlib import Path

from rag.ingest import load_sample_documents


def _write_doc(directory: Path, filename: str, content: str) -> None:
    (directory / filename).write_text(content, encoding="utf-8")


def test_load_sample_documents_produces_non_empty_chunks(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        "kart-engelleme.md",
        "# Kart Engelleme Politikası\n\n" + ("Kartınızı anında engelleyebilirsiniz. " * 40),
    )

    documents = load_sample_documents(str(tmp_path))

    assert len(documents) > 0
    assert all(document.page_content.strip() for document in documents)


def test_load_sample_documents_attaches_expected_metadata_keys(tmp_path: Path) -> None:
    _write_doc(
        tmp_path,
        "hesap-turleri.md",
        "# Hesap Türleri\n\nVadesiz ve vadeli hesap seçenekleri sunulur.",
    )

    documents = load_sample_documents(str(tmp_path))

    assert len(documents) == 1
    metadata = documents[0].metadata
    assert set(metadata.keys()) == {"doc_id", "title", "source"}
    assert metadata["doc_id"] == "hesap-turleri.md-0"
    assert metadata["source"] == "hesap-turleri.md"


def test_load_sample_documents_extracts_h1_title() -> None:
    documents = load_sample_documents("data/sample_docs")

    titles = {document.metadata["source"]: document.metadata["title"] for document in documents}
    assert titles["kart-engelleme-politikasi.md"] == "Kart Engelleme Politikası"


def test_load_sample_documents_falls_back_to_filename_when_no_h1(tmp_path: Path) -> None:
    _write_doc(tmp_path, "no-heading.md", "Bu dosyanın bir H1 başlığı yok, sadece düz metin.")

    documents = load_sample_documents(str(tmp_path))

    assert documents[0].metadata["title"] == "no-heading"


def test_load_sample_documents_chunks_long_file_into_multiple_pieces(tmp_path: Path) -> None:
    long_body = (
        "Bu cümle tekrar tekrar yazılarak dosyanın 500 karakterlik parça boyutunu aşması sağlanır. "
        * 20
    )
    _write_doc(tmp_path, "long-doc.md", f"# Uzun Belge\n\n{long_body}")

    documents = load_sample_documents(str(tmp_path))

    assert len(documents) > 1
    doc_ids = [document.metadata["doc_id"] for document in documents]
    assert doc_ids == [f"long-doc.md-{i}" for i in range(len(documents))]


def test_load_sample_documents_empty_directory_returns_empty_list(tmp_path: Path) -> None:
    assert load_sample_documents(str(tmp_path)) == []
