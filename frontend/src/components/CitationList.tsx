import { FileText } from "@phosphor-icons/react";
import type { Citation } from "@/lib/types";

interface CitationListProps {
  citations: Citation[];
}

export function CitationList({ citations }: CitationListProps) {
  if (citations.length === 0) return null;

  return (
    <div className="mt-2 space-y-1.5">
      {citations.map((citation, index) => (
        <div
          key={citation.doc_id}
          className="flex gap-2 rounded-lg border border-gray-200 bg-gray-50 p-2 text-xs"
        >
          <FileText size={14} weight="fill" className="mt-0.5 shrink-0 text-gray-400" />
          <div>
            <p className="font-medium text-gray-700">
              [{index + 1}] {citation.title}
              <span className="ml-1.5 font-normal text-gray-400">
                ({Math.round(citation.score * 100)}% eşleşme)
              </span>
            </p>
            <p className="mt-0.5 text-gray-500">{citation.snippet}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
