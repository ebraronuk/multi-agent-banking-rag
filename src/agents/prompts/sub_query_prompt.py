"""Bileşik bir mesajdan tek bir alt-niyete ait kısmı izole eden sistem promptu."""

from __future__ import annotations

SUB_QUERY_ISOLATION_SYSTEM_PROMPT = """Kullanıcının tek bir mesajında birden fazla, farklı
konudaki istek/soru var. Sana hangi konuyla ilgili kısmı ayıklaman gerektiği söylenecek.

Kurallar:
- Sadece belirtilen konuyla ilgili kısmı, kelimesi kelimesine (parafraz etmeden) döndür.
- İlgili kısmı net bir şekilde ayırt edemiyorsan mesajın tamamını olduğu gibi döndür.
- Başka hiçbir şey ekleme — açıklama, giriş cümlesi, tırnak işareti yok.
"""
