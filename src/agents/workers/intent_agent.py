"""Niyet sınıflandırmasını saran LangGraph düğümü.

Grafikte `ner_agent`'tan sonra çalışır ki sınıflandırma çıkarılmış varlıkları
kullanabilsin (bkz. `nlp/intent_classifier.py`). Ayrıca
`state["carried_pending_request"]`'in tüketicisi (ADR-008): bekleyen bir slot
bu turn dolduysa, niyet sıfırdan yeniden sınıflandırılmak yerine aynen kullanılır.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel

from agents.memory import split_cancellation
from agents.state import GraphState
from agents.supervisor import TOOL_DRIVEN_INTENTS
from nlp.intent_classifier import classify_intent
from nlp.ner_extractor import extract_entities
from nlp.text_utils import ascii_fold
from schemas.dto import AgentTraceStep, IntentLabel

# Zincirlenebilir niyetler (ADR-012). ESCALATE/OUT_OF_SCOPE dahil değil —
# bir insana aktarım isteği konuşmanın o an bittiği anlamına geliyor.
_CHAINABLE_INTENTS = frozenset(
    {*TOOL_DRIVEN_INTENTS, IntentLabel.RAG_QUERY, IntentLabel.SMALL_TALK}
)
_MAX_EXTRA_INTENTS = 2


def _clean_extra_intents(primary: IntentLabel, raw: list[IntentLabel]) -> list[IntentLabel]:
    cleaned: list[IntentLabel] = []
    for candidate in raw:
        if candidate == primary or candidate in cleaned or candidate not in _CHAINABLE_INTENTS:
            continue
        cleaned.append(candidate)
        if len(cleaned) >= _MAX_EXTRA_INTENTS:
            break
    return cleaned


# Bir önceki konuya bağlanan ifadeler. Dar tutuldu: "peki", "ya", "başka" gibi
# kelimeler tek başına bir konu açmıyor, hep bir öncekine gönderme yapıyor.
_FOLLOW_UP_MARKERS: tuple[str, ...] = (
    "peki",
    "ya ",
    "başka",
    "baska",
    "onun",
    "bunun",
    "aynısı",
    "aynisi",
    "ayrıca",
    "ayrica",
    "bir de",
)


def _looks_like_follow_up(text: str) -> bool:
    folded = ascii_fold(text.strip())
    return len(folded) <= 40 and any(
        ascii_fold(marker) in folded for marker in _FOLLOW_UP_MARKERS
    )


def _last_anchor_message(state: GraphState) -> str | None:
    """Birleştirmenin bağlanacağı son *konu açan* kullanıcı mesajı.

    Takip parçalarının atlanma sebebi ölçüldü: "eft limiti" -> "ya ücreti ne
    kadar" -> "peki hafta sonu da aynı mı" zincirinde her mesaj bir öncekine
    bağlanınca ikinci adımda "eft" çapası kayboluyor ve sorgu iki parçanın
    toplamına dönüşüyordu. Cevap da konudan çıkıp çağrı merkezi saatlerine
    kayıyordu. Çapa, kendisi takip parçası olmayan en son mesaj.
    """
    fallback: str | None = None
    for message in reversed(state.get("history", [])):
        if message.role != "user":
            continue
        if fallback is None:
            fallback = message.content
        if not _looks_like_follow_up(message.content):
            return message.content
    # Hepsi parçaysa en sonuncusuna bağlan: hiç bağlamamaktan iyi.
    return fallback


def _merged_follow_up_query(state: GraphState) -> str | None:
    """Kısa bir takip sorusunu önceki konuyla birleştirip retrieval sorgusu üretir.

    "peki ya havale" tek başına hiçbir şey getirmez, "eft limiti peki ya
    havale" doğru dokümana gider. Ham mesaj değiştirilmiyor — kullanıcının
    yazdığı şey trace'te olduğu gibi duruyor.
    """
    text = state["user_query"].strip()
    if len(text) > 40:
        return None
    folded = ascii_fold(text)
    if not any(ascii_fold(marker) in folded for marker in _FOLLOW_UP_MARKERS):
        return None
    previous = _last_anchor_message(state)
    if not previous:
        return None
    return f"{previous} {text}"


def build_intent_node(llm: BaseChatModel) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def intent_node(state: GraphState) -> dict[str, object]:
        pending = state.get("carried_pending_request")
        entities = state.get("entities", [])

        # Kullanıcı bekleyen isteği iptal ediyorsa akıştan çıkarılıyor.
        # Bu olmadan "hayır" ya da "boşver" cevabı kart numarası sanılıyor ve
        # kullanıcı başlattığı akışta kilitli kalıyordu: her mesajına
        # "bu bir kart numarası gibi görünmüyor" cevabı geliyordu.
        cancelled, remainder = split_cancellation(state["user_query"])
        if pending and cancelled and remainder:
            # "boşver bakiyeme bakayım": akış kapanıyor ama ikinci istek
            # kaybolmuyor — kalan metin normal sınıflandırmaya gidiyor.
            intent, confidence, _extra = await classify_intent(
                remainder, extract_entities(remainder), llm
            )
            return {
                "intent": intent,
                "intent_confidence": confidence,
                "extra_intents": [],
                "pending_entity_request": None,
                "user_query": remainder,
                "trace": [
                    AgentTraceStep(
                        node="intent_agent",
                        summary=f"cancelled pending {pending.intent}, continued as {intent}",
                    )
                ],
            }

        if pending and cancelled:
            return {
                "intent": IntentLabel.SMALL_TALK,
                "intent_confidence": 1.0,
                "extra_intents": [],
                "pending_entity_request": None,
                "cancelled_pending": True,
                "trace": [
                    AgentTraceStep(
                        node="intent_agent",
                        summary=f"cancelled pending {pending.intent} on user request",
                    )
                ],
            }

        if pending and any(e.type == pending.entity_type for e in entities):
            return {
                "intent": pending.intent,
                "intent_confidence": 1.0,
                "extra_intents": [],
                "trace": [
                    AgentTraceStep(
                        node="intent_agent",
                        summary=f"continued pending {pending.intent} (slot-fill answered)",
                    )
                ],
            }

        intent, confidence, extra_intents_raw = await classify_intent(
            state["user_query"], entities, llm
        )

        extra_intents = _clean_extra_intents(intent, extra_intents_raw)

        # Takip sorusu: "eft limiti" -> "peki ya havale". Bu mesaj tek başına
        # doğru dokümana gitmiyor.
        #
        # Önce yalnızca OUT_OF_SCOPE'ta devreye giriyordu, gerekçe "doğru
        # bulunmuş bir niyete dokunma" idi. Ama asıl sorun sınıflandırmada
        # değil retrieval'daydı: "ya ücreti ne kadar?" içinde "ücret" geçtiği
        # için RAG_QUERY'ye doğru şekilde düşüyor, sonra parça tek başına
        # aranınca alakasız dokümanı getiriyordu. Yani doğru niyet, yanlış
        # cevap.
        #
        # RAG_QUERY'de birleştirme yalnızca `active_sub_query`yi, yani arama
        # sorgusunu değiştiriyor; niyet, güven ve çoklu niyet olduğu gibi
        # kalıyor. Araç niyetlerinde (ACCOUNT_ACTION, CARD_ACTION) bilinçli
        # olarak çalışmıyor: orada birleştirme yönlendirmeyi bozardı.
        if intent in {IntentLabel.OUT_OF_SCOPE, IntentLabel.RAG_QUERY}:
            merged = _merged_follow_up_query(state)
            if merged is not None:
                return {
                    "intent": IntentLabel.RAG_QUERY,
                    "intent_confidence": confidence if intent is IntentLabel.RAG_QUERY else 0.4,
                    "extra_intents": extra_intents,
                    "active_sub_query": merged,
                    "trace": [
                        AgentTraceStep(
                            node="intent_agent",
                            summary="treated as follow-up to the previous question",
                        )
                    ],
                }

        summary = f"classified as {intent} (confidence={confidence:.2f})"
        if extra_intents:
            summary += f", extra: {[str(i) for i in extra_intents]}"

        return {
            "intent": intent,
            "intent_confidence": confidence,
            "extra_intents": extra_intents,
            "trace": [AgentTraceStep(node="intent_agent", summary=summary)],
        }

    return intent_node
