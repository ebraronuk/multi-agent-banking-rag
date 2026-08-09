"use client";

import { useEffect, useRef, useState } from "react";
import { PaperPlaneTilt } from "@phosphor-icons/react";
import type { ChatMessageEntry } from "@/lib/types";
import { ApiError, sendChatMessage } from "@/lib/api";
import { EXAMPLE_PROMPTS } from "@/lib/labels";
import { MessageBubble } from "./MessageBubble";
import { TraceTimeline } from "./TraceTimeline";

export function ChatPanel() {
  // Tarayıcı sekmesi oturumu başına bir id — kalıcı değil, sayfayı
  // yenilemek bilinçli olarak yeni bir konuşma başlatıyor (bu bir demo,
  // gerçek bir hesap değil; gerçek bir ürün bunu giriş yapmış bir
  // kullanıcıya bağlardı).
  const [conversationId] = useState<string>(() => crypto.randomUUID());
  const [messages, setMessages] = useState<ChatMessageEntry[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

  async function handleSend(text: string): Promise<void> {
    const trimmed = text.trim();
    if (!trimmed || isSending) return;

    setError(null);
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: trimmed }]);
    setIsSending(true);

    try {
      const response = await sendChatMessage({ conversation_id: conversationId, message: trimmed });
      setMessages((prev) => {
        const next: ChatMessageEntry[] = [
          ...prev,
          { role: "assistant", content: response.answer, response },
        ];
        setSelectedIndex(next.length - 1);
        return next;
      });
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Sunucuya ulaşılamadı — backend (localhost:8000) çalışıyor mu?",
      );
    } finally {
      setIsSending(false);
    }
  }

  const selectedResponse = selectedIndex !== null ? messages[selectedIndex]?.response : undefined;

  return (
    <div className="grid h-screen grid-cols-1 md:grid-cols-[1fr_360px]">
      <div className="flex flex-col border-r border-gray-200">
        <header className="border-b border-gray-200 px-6 py-4">
          <h1 className="text-lg font-semibold text-gray-900">DemoBank Asistanı</h1>
          <p className="text-xs text-gray-500">multi-agent-banking-rag — portföy demosu</p>
        </header>

        <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4">
          {messages.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center text-center">
              <p className="mb-4 text-sm text-gray-400">Bir örnek deneyin:</p>
              <div className="flex flex-wrap justify-center gap-2">
                {EXAMPLE_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => handleSend(prompt)}
                    className="rounded-full border border-gray-200 px-3 py-1.5 text-xs text-gray-600 transition hover:border-blue-300 hover:text-blue-600"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((entry, index) => (
            <MessageBubble
              key={index}
              entry={entry}
              isSelected={selectedIndex === index}
              onSelect={() => setSelectedIndex(index)}
            />
          ))}

          {isSending && (
            <div className="flex gap-2.5">
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-800 text-white">
                <span className="animate-pulse text-xs">···</span>
              </div>
              <div className="rounded-2xl rounded-tl-sm border border-gray-200 bg-white px-4 py-2.5 text-sm text-gray-400">
                düşünüyor…
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {error && (
          <div className="mx-6 mb-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
            {error}
          </div>
        )}

        <form
          onSubmit={(event) => {
            event.preventDefault();
            void handleSend(input);
          }}
          className="flex gap-2 border-t border-gray-200 p-4"
        >
          <input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Bir mesaj yazın…"
            className="flex-1 rounded-full border border-gray-300 px-4 py-2 text-sm focus:border-blue-400 focus:outline-none"
          />
          <button
            type="submit"
            disabled={isSending || !input.trim()}
            className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white transition disabled:opacity-40"
          >
            <PaperPlaneTilt size={18} weight="fill" />
          </button>
        </form>
        <p className="px-6 pb-3 text-[11px] text-gray-400">
          Not: insana aktarım sonrası kimlik doğrulama adımında herhangi bir 4 haneli numara
          kabul edilir (ör. 4321) — demo, gerçek bir müşteri kaydına karşı doğrulanmıyor.
        </p>
      </div>

      <aside className="hidden overflow-y-auto bg-gray-50 px-5 py-5 md:block">
        <h2 className="mb-4 text-sm font-semibold text-gray-700">Ajan İzleme (Agent Trace)</h2>
        <TraceTimeline trace={selectedResponse?.trace ?? []} />
      </aside>
    </div>
  );
}
