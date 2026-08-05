from __future__ import annotations

import tempfile

from app.core.config import Settings
from evaluation.eval_harness import run_intent_eval, run_retrieval_eval
from rag.ingest import load_sample_documents
from rag.retriever import build_retriever
from rag.vectorstore import build_vectorstore


def test_rule_based_intent_classifier_clears_a_minimum_bar_on_the_eval_set() -> None:
    result = run_intent_eval()

    assert result.accuracy >= 0.85, f"misses: {result.misses}"


def test_retrieval_finds_the_right_source_doc_at_rank_one() -> None:
    # `data/vectorstore-test` değil, özel bir geçici koleksiyon — bu test
    # kendi bilgi tabanını seed'liyor ve paylaşılan koleksiyonda başka bir
    # test çalışmasının bıraktığı şeye bağımlı olmamalı (onu da kirletmemeli).
    with tempfile.TemporaryDirectory() as tmp_dir:
        settings = Settings(chroma_persist_dir=tmp_dir, chroma_collection="eval-harness-test")
        vectorstore = build_vectorstore(settings)
        vectorstore.add_documents(load_sample_documents())

        result = run_retrieval_eval(build_retriever(settings))

    # Varsayılmadı, ölçüldü: FakeHashEmbeddings (anlam yok) + Türkçe morfolojiyi
    # kök indirgemeden BM25 tokenization'ıyla (bloklamak/bloke/blokla, düz bir
    # `.lower().split()`'a göre üç alakasız token), fake modun bu sette gerçek
    # precision@1'i ~0.5 — bu sınır o ölçüm, testi geçirmek için seçilmiş bir
    # sınır değil. Bugünkü bilinen taban çizgisinin altına bir *regresyonu*
    # yakalamak için var, offline retrieval'ın harika olduğunu iddia etmek
    # için değil; ADR-003/ADR-004 zaten production-kalitesinde retrieval için
    # gerçek bir embedding modeli gerektiğini söylüyor.
    assert result.accuracy >= 0.5, f"misses: {result.misses}"
