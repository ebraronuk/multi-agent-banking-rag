"""Deterministic, regex-based named entity recognition for banking chat turns.

Rule-based on purpose: NER here is a fast, fully auditable first pass — every
match traces back to an explicit pattern, so it's trivially unit-testable and
has zero network/latency cost on the hot path. A production system would
likely add a statistical or LLM-based NER pass alongside this one for recall
on entities that don't fit a fixed pattern (free-form person names, unusual
phrasing); this layer stays the deterministic, testable core underneath that.
"""

from __future__ import annotations

import re

from schemas.dto import Entity, EntityType

# TR IBAN: "TR" + 2 check digits + 5 groups of 4 + a final 2 = 26 characters,
# with optional single spaces between groups (how humans actually type them).
_IBAN_RE = re.compile(r"\bTR\d{2}(?:\s?\d{4}){5}\s?\d{2}\b", re.IGNORECASE)

# Amount with a currency symbol glued to the front ("€200", "$50").
# Amount with a currency code/word trailing it ("500 TRY", "200 euro").
# Two named-group sets (not one shared set) because Python's `re` forbids
# reusing a group name across alternation branches.
_AMOUNT_CURRENCY_RE = re.compile(
    r"(?P<symbol>[€$£])\s?(?P<amt1>\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?)"
    r"|"
    r"(?P<amt2>\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s?"
    r"(?P<word>TL|TRY|USD|EUR|GBP|₺|euro|dolar|dollar|lira|pound)(?![A-Za-z])",
    re.IGNORECASE,
)

_CURRENCY_ISO_MAP = {
    "TL": "TRY",
    "TRY": "TRY",
    "₺": "TRY",
    "LIRA": "TRY",
    "USD": "USD",
    "$": "USD",
    "DOLAR": "USD",
    "DOLLAR": "USD",
    "EUR": "EUR",
    "€": "EUR",
    "EURO": "EUR",
    "GBP": "GBP",
    "£": "GBP",
    "POUND": "GBP",
}

_DATE_DMY_RE = re.compile(r"\b(?P<day>\d{1,2})[./](?P<month>\d{1,2})[./](?P<year>\d{4})\b")
_DATE_YMD_RE = re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b")

# "**** 1234" / "****1234" style masking.
_CARD_MASK_RE = re.compile(r"\*{2,}\s?(?P<last4>\d{4})(?!\d)")
# A card-related keyword followed, within a short window, by exactly 4 digits
# not adjacent to other digits. The window uses `.` (not `[^\d]`) because
# Turkish phrasing like "son 4 hanesi" ("last 4 digits") embeds an unrelated
# single digit between the keyword and the real target.
_CARD_KEYWORD_RE = re.compile(r"kart\w*.{0,30}?(?<!\d)(?P<last4>\d{4})(?!\d)", re.IGNORECASE)

_ACCOUNT_TYPE_VOCAB = (
    "vadeli hesap",
    "vadesiz hesap",
    "kredi kartı",
    "döviz hesabı",
    "birikim hesabı",
)
_ACCOUNT_TYPE_PATTERNS = tuple(
    (canonical, re.compile(re.escape(canonical), re.IGNORECASE))
    for canonical in _ACCOUNT_TYPE_VOCAB
)


def _normalize_amount(raw: str) -> str:
    """Collapse Turkish (1.250,50) and plain (500 / 50,00) formats to a bare
    numeric string with '.' as the decimal separator, the shape the rest of
    the system expects (see `schemas.dto.Entity.normalized`).

    When both '.' and ',' are present, whichever appears last is treated as
    the decimal separator (handles both Turkish and US thousands/decimal
    conventions). When only one separator is present, it's treated as
    decimal only if exactly 2 digits follow it — otherwise as a thousands
    separator and stripped.
    """
    has_dot = "." in raw
    has_comma = "," in raw

    if has_dot and has_comma:
        if raw.rfind(",") > raw.rfind("."):
            integer_part, _, decimal_part = raw.replace(".", "").rpartition(",")
        else:
            integer_part, _, decimal_part = raw.replace(",", "").rpartition(".")
        return f"{integer_part}.{decimal_part}"

    if has_comma:
        integer_part, _, decimal_part = raw.rpartition(",")
        if len(decimal_part) == 2:
            return f"{integer_part}.{decimal_part}"
        return raw.replace(",", "")

    if has_dot:
        integer_part, _, decimal_part = raw.rpartition(".")
        if len(decimal_part) == 2:
            return f"{integer_part}.{decimal_part}"
        return raw.replace(".", "")

    return raw


def _normalize_currency(raw: str) -> str:
    return _CURRENCY_ISO_MAP.get(raw.upper(), raw.upper())


def _extract_ibans(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for match in _IBAN_RE.finditer(text):
        raw = match.group(0)
        entities.append(
            Entity(
                type=EntityType.IBAN,
                value=raw,
                normalized=re.sub(r"\s+", "", raw).upper(),
                start=match.start(),
                end=match.end(),
                confidence=1.0,
            )
        )
    return entities


def _extract_amounts_and_currency(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for match in _AMOUNT_CURRENCY_RE.finditer(text):
        if match.group("amt1") is not None:
            amount_raw = match.group("amt1")
            amount_start, amount_end = match.span("amt1")
            currency_raw = match.group("symbol")
            currency_start, currency_end = match.span("symbol")
        else:
            amount_raw = match.group("amt2")
            amount_start, amount_end = match.span("amt2")
            currency_raw = match.group("word")
            currency_start, currency_end = match.span("word")

        entities.append(
            Entity(
                type=EntityType.AMOUNT,
                value=amount_raw,
                normalized=_normalize_amount(amount_raw),
                start=amount_start,
                end=amount_end,
                confidence=1.0,
            )
        )
        if currency_raw:
            entities.append(
                Entity(
                    type=EntityType.CURRENCY,
                    value=currency_raw,
                    normalized=_normalize_currency(currency_raw),
                    start=currency_start,
                    end=currency_end,
                    confidence=1.0,
                )
            )
    return entities


def _extract_dates(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for match in _DATE_DMY_RE.finditer(text):
        day, month, year = match.group("day"), match.group("month"), match.group("year")
        entities.append(
            Entity(
                type=EntityType.DATE,
                value=match.group(0),
                normalized=f"{year}-{int(month):02d}-{int(day):02d}",
                start=match.start(),
                end=match.end(),
                confidence=1.0,
            )
        )
    for match in _DATE_YMD_RE.finditer(text):
        entities.append(
            Entity(
                type=EntityType.DATE,
                value=match.group(0),
                normalized=match.group(0),
                start=match.start(),
                end=match.end(),
                confidence=1.0,
            )
        )
    return entities


def _extract_card_last4(text: str) -> list[Entity]:
    entities: list[Entity] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern in (_CARD_MASK_RE, _CARD_KEYWORD_RE):
        for match in pattern.finditer(text):
            span = match.span("last4")
            if span in seen_spans:
                continue
            seen_spans.add(span)
            last4 = match.group("last4")
            entities.append(
                Entity(
                    type=EntityType.CARD_LAST4,
                    value=last4,
                    normalized=last4,
                    start=span[0],
                    end=span[1],
                    confidence=1.0,
                )
            )
    return entities


def _extract_account_types(text: str) -> list[Entity]:
    entities: list[Entity] = []
    for canonical, pattern in _ACCOUNT_TYPE_PATTERNS:
        for match in pattern.finditer(text):
            entities.append(
                Entity(
                    type=EntityType.ACCOUNT_TYPE,
                    value=match.group(0),
                    normalized=canonical,
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0,
                )
            )
    return entities


def extract_entities(text: str) -> list[Entity]:
    entities: list[Entity] = [
        *_extract_ibans(text),
        *_extract_amounts_and_currency(text),
        *_extract_dates(text),
        *_extract_card_last4(text),
        *_extract_account_types(text),
    ]
    # Sort by position so trace/API output reads left-to-right like the
    # source message, regardless of which sub-extractor found what.
    entities.sort(key=lambda entity: entity.start if entity.start is not None else 0)
    return entities
