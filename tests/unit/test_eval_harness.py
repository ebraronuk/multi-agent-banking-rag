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
    # A dedicated temp collection, not `data/vectorstore-test` — this test
    # seeds its own knowledge base and shouldn't depend on (or pollute)
    # whatever another test run left behind in the shared one.
    with tempfile.TemporaryDirectory() as tmp_dir:
        settings = Settings(chroma_persist_dir=tmp_dir, chroma_collection="eval-harness-test")
        vectorstore = build_vectorstore(settings)
        vectorstore.add_documents(load_sample_documents())

        result = run_retrieval_eval(build_retriever(settings))

    # Measured, not assumed: with FakeHashEmbeddings (no semantics) + BM25
    # tokenization that doesn't stem Turkish morphology (bloklamak/bloke/
    # blokla are three unrelated tokens to a plain `.lower().split()`), fake
    # mode's real precision@1 on this set is ~0.5 — this bar is that
    # measurement, not a bar picked to make the test pass. It exists to catch
    # a *regression* below today's known baseline, not to claim retrieval is
    # great offline; ADR-003/ADR-004 already say a real embedding model is
    # what production-quality retrieval needs.
    assert result.accuracy >= 0.5, f"misses: {result.misses}"
