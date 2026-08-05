import type { Icon } from "@phosphor-icons/react";
import {
  Brain,
  ChatCircleDots,
  Database,
  FloppyDisk,
  MagnifyingGlass,
  ShieldCheck,
  TreeStructure,
  UserSwitch,
  BookOpen,
  Wrench,
} from "@phosphor-icons/react";

// One config object driving the trace timeline's icon/label/color per node —
// keeps agents/graph.py's node names as the single source of truth instead of
// scattering `if (node === "rag_agent")` branches across components.
export interface NodeMeta {
  label: string;
  icon: Icon;
  colorClass: string; // Tailwind text/bg color pair, applied together
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
