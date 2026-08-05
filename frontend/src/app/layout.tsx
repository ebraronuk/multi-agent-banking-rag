import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "DemoBank Asistanı — multi-agent-banking-rag",
  description:
    "LangGraph ile orkestre edilmiş, RAG + araç çağırma + konuşma hafızasına sahip bir bankacılık destek asistanı demosu.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="tr" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex h-full min-h-full flex-col">{children}</body>
    </html>
  );
}
