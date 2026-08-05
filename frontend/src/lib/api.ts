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
    // The backend always returns this shape for errors (see
    // app/main.py's exception handlers) — 422/429/500 all match ErrorResponse.
    const body = (await response.json().catch(() => null)) as ApiErrorResponse | null;
    throw new ApiError(
      body?.message ?? "Beklenmeyen bir hata oluştu.",
      body?.code ?? "UNKNOWN",
      response.status,
    );
  }

  return (await response.json()) as ChatResponse;
}
