import { ShieldWarning } from "@phosphor-icons/react";
import type { GuardrailFlag } from "@/lib/types";
import { GUARDRAIL_FLAG_LABELS } from "@/lib/labels";

interface GuardrailBadgesProps {
  flags: GuardrailFlag[];
}

export function GuardrailBadges({ flags }: GuardrailBadgesProps) {
  if (flags.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-1.5">
      {flags.map((flag) => (
        <span
          key={flag}
          className="inline-flex items-center gap-1 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-xs font-medium text-amber-700"
        >
          <ShieldWarning size={12} weight="fill" />
          {GUARDRAIL_FLAG_LABELS[flag]}
        </span>
      ))}
    </div>
  );
}
