import { FileText } from "@phosphor-icons/react";
import type { Citation } from "@/lib/types";

interface CitationListProps {
  citations: Citation[];
}

// `score`, bu sorgunun döndürdüğü adaylar arasında min-max normalize edilmiş
// bağıl bir sıralama — mutlak bir güven ölçüsü değil (bkz. rag/reranker.py).
// Bunu doğrudan "%86 eşleşme" gibi göstermek, zayıf/alakasız bir sorguda bile
// yüksek görünen bir sayı vererek yanıltıcı bir kesinlik izlenimi verirdi;
// üç kaba katmana indirip ham sayıyı sadece tooltip'te (bağlamıyla) tutuyoruz.
function matchTier(score: number): { label: string; className: string } {
  if (score >= 0.7) return { label: "Güçlü eşleşme", className: "text-emerald-600" };
  if (score >= 0.4) return { label: "Orta düzey eşleşme", className: "text-amber-600" };
  return { label: "Zayıf eşleşme", className: "text-gray-400" };
}

export function CitationList({ citations }: CitationListProps) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-2 space-y-1.5">
      {citations.map((citation, index) => {
        const tier = matchTier(citation.score);
        return (
          <div
            key={citation.doc_id}
            className="flex gap-2 rounded-lg border border-gray-200 bg-gray-50 p-2 text-xs"
          >
            <FileText size={14} weight="fill" className="mt-0.5 shrink-0 text-gray-400" />
            <div>
              <p className="font-medium text-gray-700">
                [{index + 1}] {citation.title}
                <span
                  className={`ml-1.5 font-normal ${tier.className}`}
                  title={`Bu sorgunun diğer adaylarına göre bağıl skor: ${Math.round(citation.score * 100)}%`}
                >
                  ({tier.label})
                </span>
              </p>
              <p className="mt-0.5 text-gray-500">{citation.snippet}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
