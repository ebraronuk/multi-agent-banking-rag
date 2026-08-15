"""nlp/text_utils.py için birim testler."""

from __future__ import annotations

from nlp.text_utils import turkish_lower


def test_turkish_capital_i_dotted_lowers_to_plain_ascii_i() -> None:
    assert turkish_lower("İyi akşamlar") == "iyi akşamlar"


def test_regular_lowercasing_still_works() -> None:
    assert turkish_lower("MERHABA World") == "merhaba world"


def test_substring_match_survives_capital_i_dotted_prefix() -> None:
    assert "iyi akşam" in turkish_lower("İyi akşamlar, nasılsınız?")
