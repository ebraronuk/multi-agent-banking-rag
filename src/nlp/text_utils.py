"""Küçük, paylaşılan metin normalizasyon yardımcıları."""

from __future__ import annotations

# Türkçe harf -> ASCII karşılığı. `unicodedata.normalize("NFKD", ...)` ile
# yapılmıyor çünkü "ı" (dotless i) NFKD altında ayrışmıyor, olduğu gibi
# kalıyor — Türkçe için hazır Unicode yolu eksik çalışıyor, elle tablo
# gerekiyor.
_ASCII_FOLD = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "ş": "s",
        "Ş": "s",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
        "â": "a",
        "Â": "a",
        "î": "i",
        "Î": "i",
        "û": "u",
        "Û": "u",
    }
)


def turkish_lower(text: str) -> str:
    """`str.lower()` Türkçe "İ"yi "i"+combining-dot'a çevirip alt-dize
    eşleşmesini sessizce bozuyor — "İ"yi önce düz "i"ye çeviriyoruz."""
    return text.replace("İ", "i").lower()


def ascii_fold(text: str) -> str:
    """Türkçe harfleri ASCII karşılıklarına indirger ve küçük harfe çevirir.

    Neden gerekli: Türkiye'de kullanıcıların önemli bir kısmı Türkçe klavye
    kullanmıyor ya da kullanmayı bırakıyor. "müşteri" yerine "musteri",
    "görmek" yerine "gormek", "hesabımda" yerine "hesabimda" yazıyorlar.
    Anahtar kelime listeleri doğru Türkçe ile yazıldığı için bu mesajlar
    hiçbir kelimeyi tutturamıyor ve OUT_OF_SCOPE'a düşüyorlar.

    Ölçülen etki (bkz. `data/eval/real_user_utterances.json`): Türkçe
    karaktersiz girdide niyet doğruluğu %22'den yukarı çıkıyor; bu sınıf,
    gerçek kullanıcı dili setindeki en büyük tek hata kaynağıydı.

    `turkish_lower` ile ilişkisi: o, "İ" özelinde Python'ın bozuk davranışını
    düzeltiyor ve metnin okunabilirliğini koruyor. Bu ise **yalnızca
    eşleştirme için** kullanılacak, kayıplı bir indirgeme — kullanıcıya geri
    gösterilen metinde asla kullanılmamalı, çünkü "sıkı" ile "şıkı" aynı
    dizeye düşüyor.
    """
    return turkish_lower(text).translate(_ASCII_FOLD)
