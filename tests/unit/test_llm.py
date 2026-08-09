"""`app/core/llm.py::content_to_text` için birim testler.

Canlı doğrulanmış bir regresyon: `gemini-flash-latest` gibi bazı modeller
`AIMessage.content`'i düz `str` değil, içerik-bloğu listesi olarak dönüyor
(`[{"type": "text", "text": "...", "extras": {...}}]`) — buna körü körüne
`str()` çağırmak kullanıcıya ham bir Python repr'i gösteriyordu.
"""

from __future__ import annotations

from app.core.llm import content_to_text


def test_content_to_text_passes_plain_string_through() -> None:
    assert content_to_text("merhaba") == "merhaba"


def test_content_to_text_extracts_text_from_single_content_block() -> None:
    content = [{"type": "text", "text": "merhaba", "extras": {"signature": "abc"}}]
    assert content_to_text(content) == "merhaba"


def test_content_to_text_joins_multiple_text_blocks() -> None:
    content = [{"type": "text", "text": "merhaba, "}, {"type": "text", "text": "nasılsınız?"}]
    assert content_to_text(content) == "merhaba, nasılsınız?"


def test_content_to_text_skips_non_text_blocks() -> None:
    content = [
        {"type": "thinking", "thinking": "önce şunu düşünmeliyim"},
        {"type": "text", "text": "sonuç bu"},
    ]
    assert content_to_text(content) == "sonuç bu"


def test_content_to_text_handles_plain_strings_inside_list() -> None:
    assert content_to_text(["merhaba"]) == "merhaba"


def test_content_to_text_returns_empty_string_for_empty_list() -> None:
    assert content_to_text([]) == ""
