import type { Icon } from "@phosphor-icons/react";
import {
  ArrowRight,
  Brain,
  ChatCircleDots,
  Database,
  FloppyDisk,
  MagnifyingGlass,
  ShieldCheck,
  Stack,
  TreeStructure,
  UserSwitch,
  BookOpen,
  Wrench,
} from "@phosphor-icons/react";

// Trace timeline'ın düğüm başına ikon/etiket/renk'ini süren tek bir config
// nesnesi — agents/graph.py'nin düğüm adlarını tek kaynak olarak tutuyor,
// bileşenlere yayılmış `if (node === "rag_agent")` dallarını önlüyor.
export interface NodeMeta {
  label: string;
  icon: Icon;
  colorClass: string; // Tailwind text/bg renk çifti, birlikte uygulanıyor
}

export const NODE_META: Record<string, NodeMeta> = {
  memory_load: {
    label: "Hafıza Yüklendi",
    icon: Database,
    colorClass: "bg-slate-100 text-slate-600 border-slate-200",
  },
  ner_agent: {
    label: "Varlık Çıkarımı",
    icon: MagnifyingGlass,
    colorClass: "bg-sky-50 text-sky-700 border-sky-200",
  },
  intent_agent: {
    label: "Niyet Sınıflandırma",
    icon: Brain,
    colorClass: "bg-violet-50 text-violet-700 border-violet-200",
  },
  supervisor: {
    label: "Supervisor Yönlendirme",
    icon: TreeStructure,
    colorClass: "bg-amber-50 text-amber-700 border-amber-200",
  },
  rag_agent: {
    label: "RAG — Bilgi Tabanı",
    icon: BookOpen,
    colorClass: "bg-emerald-50 text-emerald-700 border-emerald-200",
  },
  tool_agent: {
    label: "Araç Çağırma (MCP)",
    icon: Wrench,
    colorClass: "bg-orange-50 text-orange-700 border-orange-200",
  },
  advance_intent: {
    label: "Sıradaki Niyet",
    icon: ArrowRight,
    colorClass: "bg-indigo-50 text-indigo-700 border-indigo-200",
  },
  synthesizer: {
    label: "Yanıt Birleştirme",
    icon: Stack,
    colorClass: "bg-fuchsia-50 text-fuchsia-700 border-fuchsia-200",
  },
  smalltalk: {
    label: "Sohbet",
    icon: ChatCircleDots,
    colorClass: "bg-pink-50 text-pink-700 border-pink-200",
  },
  escalate: {
    label: "İnsana Aktarım",
    icon: UserSwitch,
    colorClass: "bg-rose-50 text-rose-700 border-rose-200",
  },
  guardrail: {
    label: "Güvenlik Kontrolü",
    icon: ShieldCheck,
    colorClass: "bg-teal-50 text-teal-700 border-teal-200",
  },
  memory_save: {
    label: "Hafıza Kaydedildi",
    icon: FloppyDisk,
    colorClass: "bg-slate-100 text-slate-600 border-slate-200",
  },
};

export const DEFAULT_NODE_META: NodeMeta = {
  label: "Bilinmeyen Adım",
  icon: Database,
  colorClass: "bg-gray-50 text-gray-600 border-gray-200",
};

export function getNodeMeta(node: string): NodeMeta {
  return NODE_META[node] ?? DEFAULT_NODE_META;
}
