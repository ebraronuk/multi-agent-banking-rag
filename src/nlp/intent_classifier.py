"""Rule-based and LLM-backed intent classification.

The rule-based path is always available (offline, deterministic, no network)
and is what `classify_intent` falls back to whenever the LLM path can't be
trusted: the process is running on `FakeChatModel` (a hash digest has no
notion of "which intent applies here", so prompting it would be theater), or
the real LLM call fails or returns something that doesn't parse. An intent
classifier that raises because a provider returned malformed structured
output would take down the whole conversational turn — that's a real
production bug class, guarded against explicitly rather than left implicit.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.prompts.intent_prompt import INTENT_SYSTEM_PROMPT
from app.core.llm import is_fake_model
from app.core.logging import get_logger
from schemas.dto import Entity, EntityType, IntentLabel

logger = get_logger(__name__)

# Dict order doubles as the tie-break rule: `classify_intent_rule_based`'s
# `max()` keeps the first-seen top score, so RAG_QUERY (listed first) wins any
# tie against an action intent. This matters concretely for "EFT limitiniz ne
# kadar?" — it matches both "limit" (RAG_QUERY) and "eft" (TRANSACTION_ACTION);
# defaulting an ambiguous message to "explain" rather than "act" is the safer
# failure mode for a banking assistant (nothing gets moved/blocked on a guess).
_INTENT_KEYWORDS: dict[IntentLabel, tuple[str, ...]] = {
    IntentLabel.RAG_QUERY: (
        "nasıl açılır",
        "nasıl yapılır",
        "nasıl kullanılır",
        "nasıl başvur",
        "politika",
        "ücret",
        "komisyon",
        "çalışma saat",
        "ne kadar sürer",
        "şart",
        "koşul",
        "limit",
        "how do i",
        "policy",
        "fee",
        "what is the",
    ),
    IntentLabel.ACCOUNT_ACTION: (
        "bakiye",
        "hesap özeti",
        "hesabım ne kadar",
        "hesap hareketleri",
        "balance",
        "account summary",
        "my balance",
    ),
    IntentLabel.TRANSACTION_ACTION: (
        "işlem geçmişi",
        "son işlem",
        "para transfer",
        "havale",
        "eft",
        "harcama",
        "para gönder",
        "transaction history",
        "transfer",
        "send money",
    ),
    IntentLabel.CARD_ACTION: (
        "kartımı blokla",
        "kart engelle",
        "bloke",
        "kartım çalındı",
        "kartımı çaldılar",
        "kartımı kaybettim",
        "kartım kayboldu",
        "kartımı iptal",
        "block my card",
        "card stolen",
        "lost my card",
    ),
    IntentLabel.SMALL_TALK: (
        "merhaba",
        "selam",
        "teşekkür",
        "nasılsın",
        "günaydın",
        "iyi günler",
        "hello",
        "thanks",
        "thank you",
        "how are you",
    ),
    IntentLabel.ESCALATE: (
        "temsilciyle görüş",
        "insana bağla",
        "müşteri temsilcisi",
        "gerçek bir kişi",
        "speak to a human",
        "representative",
        "human agent",
    ),
    # IntentLabel.OUT_OF_SCOPE deliberately has no keyword list — it's the
    # fallback when every other intent scores zero, not something to match for.
}

# A corroborating entity type is a modest, additive signal on top of keyword
# evidence — not a substitute for it — so it adds one flat point rather than
# a multiplier or a dominant term.
_ENTITY_BOOSTS: dict[IntentLabel, tuple[EntityType, ...]] = {
    IntentLabel.CARD_ACTION: (EntityType.CARD_LAST4,),
    IntentLabel.ACCOUNT_ACTION: (EntityType.IBAN, EntityType.ACCOUNT_TYPE),
    IntentLabel.TRANSACTION_ACTION: (EntityType.IBAN, EntityType.AMOUNT),
}

_ENTITY_BOOST_WEIGHT = 1
_CONFIDENCE_BASE = 0.4
_CONFIDENCE_PER_POINT = 0.15
_CONFIDENCE_CAP = 0.95  # a rule-based classifier should never claim full certainty
_OUT_OF_SCOPE_CONFIDENCE = 0.3


class _IntentClassification(BaseModel):
    """Internal parsing target for `llm.with_structured_output` — not a public DTO."""

    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)


def classify_intent_rule_based(text: str, entities: list[Entity]) -> tuple[IntentLabel, float]:
    lowered = text.lower()
    entity_types = {entity.type for entity in entities}

    scores: dict[IntentLabel, int] = {
        intent: sum(1 for keyword in keywords if keyword in lowered)
        for intent, keywords in _INTENT_KEYWORDS.items()
    }
    for intent, boost_types in _ENTITY_BOOSTS.items():
        if any(entity_type in entity_types for entity_type in boost_types):
            scores[intent] = scores.get(intent, 0) + _ENTITY_BOOST_WEIGHT

    best_intent = max(scores, key=lambda intent: scores[intent])
    best_score = scores[best_intent]
    if best_score <= 0:
        return IntentLabel.OUT_OF_SCOPE, _OUT_OF_SCOPE_CONFIDENCE

    confidence = min(_CONFIDENCE_CAP, _CONFIDENCE_BASE + _CONFIDENCE_PER_POINT * best_score)
    return best_intent, confidence


async def classify_intent(
    text: str, entities: list[Entity], llm: BaseChatModel
) -> tuple[IntentLabel, float]:
    if is_fake_model(llm):
        # Prompting a hash-based fake model buys nothing — it can't honor a
        # structured-output contract in any meaningful sense.
        return classify_intent_rule_based(text, entities)

    try:
        structured_llm = llm.with_structured_output(_IntentClassification)
        result = await structured_llm.ainvoke(
            [
                SystemMessage(content=INTENT_SYSTEM_PROMPT),
                HumanMessage(content=text),
            ]
        )
        if not isinstance(result, _IntentClassification):
            raise TypeError(f"unexpected structured output type: {type(result)!r}")
        return result.intent, result.confidence
    except Exception:
        logger.warning("intent_llm_classification_failed", text_preview=text[:120], exc_info=True)
        return classify_intent_rule_based(text, entities)
