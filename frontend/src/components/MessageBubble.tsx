import { Robot, User } from "@phosphor-icons/react";
import type { ChatMessageEntry } from "@/lib/types";
import { INTENT_LABELS } from "@/lib/labels";
import { CitationList } from "./CitationList";
import { ToolCallList } from "./ToolCallList";
import { GuardrailBadges } from "./GuardrailBadges";

interface MessageBubbleProps {
  entry: ChatMessageEntry;
  isSelected: boolean;
  onSelect: () => void;
}

export function MessageBubble({ entry, isSelected, onSelect }: MessageBubbleProps) {
  const isUser = entry.role === "user";

  return (
    <div className={`flex gap-2.5 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
          isUser ? "bg-blue-600 text-white" : "bg-slate-800 text-white"
        }`}
      >
        {isUser ? <User size={16} weight="fill" /> : <Robot size={16} weight="fill" />}
      </div>

      <div className={`max-w-[75%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        <button
          type="button"
          onClick={onSelect}
          disabled={!entry.response}
          className={`rounded-2xl px-4 py-2.5 text-left text-sm leading-relaxed transition ${
            isUser
              ? "rounded-tr-sm bg-blue-600 text-white"
              : `rounded-tl-sm border bg-white text-gray-800 ${
                  isSelected && entry.response
                    ? "border-blue-300 ring-2 ring-blue-100"
                    : "border-gray-200"
                } ${entry.response ? "cursor-pointer hover:border-blue-200" : ""}`
          }`}
        >
          {entry.content}
        </button>

        {entry.response && (
          <div className="mt-1.5 w-full">
            <span className="text-[11px] font-medium tracking-wide text-gray-400 uppercase">
              {INTENT_LABELS[entry.response.intent]}
            </span>
            <CitationList citations={entry.response.citations} />
            <ToolCallList toolCalls={entry.response.tool_calls} />
            {entry.response.guardrail_flags.length > 0 && (
              <div className="mt-1.5">
                <GuardrailBadges flags={entry.response.guardrail_flags} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
