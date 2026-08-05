import type { GuardrailFlag, IntentLabel } from "./types";

export const INTENT_LABELS: Record<IntentLabel, string> = {
  RAG_QUERY: "Bilgi Sorgusu",
  ACCOUNT_ACTION: "Hesap İşlemi",
  TRANSACTION_ACTION: "İşlem Geçmişi",
  CARD_ACTION: "Kart İşlemi",
  SMALL_TALK: "Sohbet",
  ESCALATE: "İnsana Aktarım",
  OUT_OF_SCOPE: "Kapsam Dışı",
};

export const GUARDRAIL_FLAG_LABELS: Record<GuardrailFlag, string> = {
  PII_REDACTED: "Kişisel veri gizlendi",
  FINANCIAL_ADVICE_BLOCKED: "Yatırım tavsiyesi engellendi",
  ESCALATED_ITERATION_LIMIT: "İterasyon sınırına ulaşıldı",
  NO_DRAFT_PRODUCED: "Yanıt üretilemedi",
};

export const EXAMPLE_PROMPTS: string[] = [
  "EFT limitiniz ne kadar?",
  "Kartımı blokla",
  "Merhaba, nasılsınız?",
  "Bir müşteri temsilcisiyle görüşmek istiyorum",
];
