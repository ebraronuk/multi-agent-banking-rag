"""`python -m evaluation` — değerlendirmeyi çalıştır, raporla, kapıyı uygula.

Kullanım:

    python -m evaluation                # ölç ve yazdır
    python -m evaluation --save         # sonucu yeni temel olarak kaydet
    python -m evaluation --gate         # temele karşı ölç, regresyonda exit 1
    python -m evaluation --gate --push  # ayrıca Langfuse'a skor gönder

`--gate` CI için: bir prompt ya da retrieval değişikliği doğruluğu düşürdüğünde
ya da p95'i yarıdan fazla artırdığında build kırılır. Kapının değeri burada —
bir değerlendirme ancak bir şeyi engelleyebiliyorsa regresyon yakalar.
"""

from __future__ import annotations

import argparse
import sys
import tempfile

from evaluation.real_utterances import run_real_utterance_eval
from evaluation.report import compare_to_baseline, push_to_langfuse, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation")
    parser.add_argument("--save", action="store_true", help="sonucu yeni temel olarak kaydet")
    parser.add_argument("--gate", action="store_true", help="regresyon varsa exit 1")
    parser.add_argument("--push", action="store_true", help="skorları Langfuse'a gönder")
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="yalnızca intent ölç (vektör deposu kurmadan, hızlı)",
    )
    args = parser.parse_args(argv)

    if args.no_retrieval:
        report = run_all(retriever=None)
    else:
        from app.core.config import get_settings
        from rag.ingest import load_sample_documents
        from rag.retriever import build_retriever
        from rag.vectorstore import build_vectorstore

        # Geçici koleksiyon: `add_documents` dedup yapmıyor, canlı koleksiyona
        # karşı ölçmek her koşuda chunk biriktirir ve skoru kaydırır.
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_settings = get_settings().model_copy(
                update={"chroma_persist_dir": tmp_dir, "chroma_collection": "eval-report-cli"}
            )
            vectorstore = build_vectorstore(eval_settings)
            vectorstore.add_documents(load_sample_documents())
            report = run_all(retriever=build_retriever(eval_settings))

    print(report.render())

    # Gerçek kullanıcı dili ayrı raporlanıyor: temiz set neredeyse her zaman
    # %100 veriyor ve tek başına yanıltıcı. Asıl bilgi, ikisi arasındaki fark.
    real = run_real_utterance_eval()
    if real.metric.total:
        print(real.render())
        report.metrics.append(real.metric)

    if args.push:
        print("Langfuse'a gönderildi." if push_to_langfuse(report) else "Langfuse kapalı, gönderilmedi.")

    if args.gate:
        regressions = compare_to_baseline(report)
        if regressions:
            print("\nREGRESYON TESPİT EDİLDİ:")
            for regression in regressions:
                print(f"  - {regression}")
            return 1
        print("\nRegresyon yok.")

    if args.save:
        path = report.save()
        print(f"\nTemel kaydedildi: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
