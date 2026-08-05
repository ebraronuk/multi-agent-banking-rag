import type { AgentTraceStep } from "@/lib/types";
import { getNodeMeta } from "@/lib/nodeMeta";

interface TraceTimelineProps {
  trace: AgentTraceStep[];
}

/**
 * The system's differentiating feature made visible: which agent ran, in
 * what order, doing what — the same `trace` field the API returns for every
 * turn (see docs/architecture.md), just rendered instead of read as JSON.
 */
export function TraceTimeline({ trace }: TraceTimelineProps) {
  if (trace.length === 0) {
    return (
      <p className="text-sm text-gray-400 italic">
        Bir mesaj gönderin, ajan zinciri burada canlanacak.
      </p>
    );
  }

  return (
    <ol className="space-y-0">
      {trace.map((step, index) => {
        const meta = getNodeMeta(step.node);
        const Icon = meta.icon;
        const isLast = index === trace.length - 1;

        return (
          <li key={`${step.node}-${index}`} className="flex gap-3">
            <div className="flex flex-col items-center">
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-full border ${meta.colorClass}`}
              >
                <Icon size={18} weight="fill" />
              </div>
              {!isLast && <div className="w-px flex-1 bg-gray-200" />}
            </div>
            <div className={isLast ? "pb-1" : "pb-5"}>
              <p className="text-sm font-medium text-gray-800">{meta.label}</p>
              <p className="text-xs text-gray-500">{step.summary}</p>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
