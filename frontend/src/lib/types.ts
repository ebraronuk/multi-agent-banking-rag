// Mirrors src/schemas/dto.py exactly — this is the one place the frontend's
// idea of the API contract lives. If the backend's DTOs change, this is what
// needs updating (no shared codegen here on purpose: a small, hand-written
// frontend doesn't need an OpenAPI client generation step to stay honest).

export type IntentLabel =
  | "RAG_QUERY"
  | "ACCOUNT_ACTION"
  | "TRANSACTION_ACTION"
  | "CARD_ACTION"
  | "SMALL_TALK"
  | "ESCALATE"
  | "OUT_OF_SCOPE";

export type EntityType =
  | "IBAN"
  | "AMOUNT"
  | "CURRENCY"
  | "DATE"
  | "CARD_LAST4"
  | "ACCOUNT_TYPE"
  | "PERSON_NAME";

export type GuardrailFlag =
  | "PII_REDACTED"
  | "FINANCIAL_ADVICE_BLOCKED"
  | "ESCALATED_ITERATION_LIMIT"
  | "NO_DRAFT_PRODUCED";

export interface Entity {
  type: EntityType;
  value: string;
  normalized: string | null;
  start: number | null;
  end: number | null;
  confidence: number;
}

export interface Citation {
  doc_id: string;
  title: string;
  source: string;
  snippet: string;
  score: number;
}

export interface ToolCallRecord {
  tool_name: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown> | string | null;
  ok: boolean;
  error: string | null;
  latency_ms: number;
}

export interface AgentTraceStep {
  node: string;
  summary: string;
  metadata: Record<string, unknown>;
}

export interface ChatRequest {
  conversation_id?: string | null;
  message: string;
}

export interface ChatResponse {
  conversation_id: string;
  answer: string;
  intent: IntentLabel;
  entities: Entity[];
  citations: Citation[];
  tool_calls: ToolCallRecord[];
  trace: AgentTraceStep[];
  guardrail_flags: GuardrailFlag[];
  iterations: number;
}

export interface ApiErrorResponse {
  code: string;
  message: string;
  details?: Record<string, unknown> | null;
}

export interface ChatMessageEntry {
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse; // present for assistant entries — drives the trace panel
}
