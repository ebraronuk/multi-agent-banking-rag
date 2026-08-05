import type { ApiErrorResponse, ChatRequest, ChatResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    // Backend hatalar için her zaman bu şekli döner (bkz. app/main.py'nin
    // exception handler'ları) — 422/429/500 hepsi ErrorResponse ile eşleşiyor.
    const body = (await response.json().catch(() => null)) as ApiErrorResponse | null;
    throw new ApiError(
      body?.message ?? "Beklenmeyen bir hata oluştu.",
      body?.code ?? "UNKNOWN",
      response.status,
    );
  }

  return (await response.json()) as ChatResponse;
}
