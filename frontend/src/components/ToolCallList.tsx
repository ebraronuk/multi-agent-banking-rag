import { CheckCircle, Wrench, XCircle } from "@phosphor-icons/react";
import type { ToolCallRecord } from "@/lib/types";

interface ToolCallListProps {
  toolCalls: ToolCallRecord[];
}

export function ToolCallList({ toolCalls }: ToolCallListProps) {
  if (toolCalls.length === 0) return null;

  return (
    <div className="mt-2 space-y-1.5">
      {toolCalls.map((call, index) => (
        <div
          key={`${call.tool_name}-${index}`}
          className="flex items-center gap-2 rounded-lg border border-orange-200 bg-orange-50 px-2 py-1.5 text-xs text-orange-800"
        >
          <Wrench size={14} weight="fill" className="shrink-0" />
          <span className="font-mono font-medium">{call.tool_name}</span>
          {call.ok ? (
            <CheckCircle size={14} weight="fill" className="ml-auto shrink-0 text-emerald-600" />
          ) : (
            <XCircle size={14} weight="fill" className="ml-auto shrink-0 text-rose-500" />
          )}
        </div>
      ))}
    </div>
  );
}
