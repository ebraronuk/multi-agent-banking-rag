"""Küçük, paylaşılan metin normalizasyon yardımcıları."""

from __future__ import annotations


def turkish_lower(text: str) -> str:
    """`str.lower()` Türkçe "İ"yi "i"+combining-dot'a çevirip alt-dize
    eşleşmesini sessizce bozuyor — "İ"yi önce düz "i"ye çeviriyoruz."""
    return text.replace("İ", "i").lower()
