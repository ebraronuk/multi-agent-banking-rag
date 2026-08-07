// src/schemas/dto.py'yi birebir yansıtıyor — frontend'in API sözleşmesi
// hakkındaki fikrinin yaşadığı tek yer burası. Backend'in DTO'ları
// değişirse güncellenmesi gereken yer burası (bilinçli olarak paylaşılan bir
// codegen yok: küçük, elle yazılmış bir frontend'in dürüst kalmak için bir
// OpenAPI client üretim adımına ihtiyacı yok).

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
  | "NO_DRAFT_PRODUCED"
  | "PROMPT_INJECTION_DETECTED"
  | "MODEL_IDENTITY_REDACTED";

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
  response?: ChatResponse; // asistan girdileri için var — trace panelini besliyor
}
