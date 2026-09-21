"""Kural tabanlı ve LLM destekli niyet sınıflandırması.

Kural tabanlı yol offline/deterministik ve `classify_intent`'in LLM yoluna
güvenilemediği her durumda (fake model, sağlayıcı hatası, parse edilemeyen
çıktı) düştüğü yer — süreç bu yüzden raise etmez.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.prompts.intent_prompt import INTENT_SYSTEM_PROMPT
from app.core.llm import is_fake_model
from app.core.logging import get_logger
from nlp.text_utils import ascii_fold
from schemas.dto import Entity, EntityType, IntentLabel

logger = get_logger(__name__)

# Dict sırası berabere-bozma kuralı: `max()` ilk-görüleni koruyor, yani
# RAG_QUERY (ilk listelenen) action intent'lere karşı beraberlikleri kazanır —
# belirsiz bir mesajda "işlem yap" yerine "açıkla"yı varsaymak daha güvenli.
# Deterministik bir fallback (ADR-003), recall'ı sınırlı (bkz. test_intent_classifier.py).
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
        "üst sınır",
        "azami",
        "en fazla",
        "how do i",
        "policy",
        "fee",
        "what is the",
    ),
    # Türkçe sondan eklemeli: "hesap" -> "hesabım", "hesabımda", "hesabımdaki".
    # Çekimli tam biçim listelemek her eki tek tek yazmayı gerektiriyor ve
    # kaçınılmaz olarak eksik kalıyor; bunun yerine ekin önündeki kök
    # yazılıyor ve alt-dize eşleşmesi çekimleri kendiliğinden yakalıyor.
    IntentLabel.ACCOUNT_ACTION: (
        "bakiye",
        "bakiyem",
        "hesap özeti",
        "hesabım",
        "hesap durum",
        "param",
        "balance",
        "account summary",
        "my balance",
    ),
    # Sadece görüntüleme ifadeleri — para gönderme niyeti taşıyan kalıplar
    # bilinçli olarak burada değil, ESCALATE'te (tool_agent parayı gerçekten
    # göndürmüyor; burada sınıflandırılırsa "yapıldı" izlenimi verirdi).
    IntentLabel.TRANSACTION_ACTION: (
        "işlem geçmişi",
        "son işlem",
        # "harca" kökü: harcama, harcadım, harcamışım, harcamalarım.
        "harca",
        # "hesap hareketleri" daha önce ACCOUNT_ACTION'daydı. Türkçe bankacılık
        # dilinde bu ifade işlem geçmişi demek, hesap özeti değil — gerçek
        # kullanıcı dili setinde "hesap hareketlerimi görmek istiyorum"
        # yanlış kovaya düşünce fark edildi.
        "hesap hareket",
        "transaction history",
    ),
    IntentLabel.CARD_ACTION: (
        "kartımı blokla",
        "kart engelle",
        "bloke",
        "dondur",
        "engelle",
        "izinsiz işlem",
        "şüpheli işlem",
        "kartım çalındı",
        "kartımı çaldılar",
        "kartımı kaybettim",
        "kartım kayboldu",
        "kartımı iptal",
        # Çıplak "kartım" kökü bilinçli olarak YOK. Denendi ve geri alındı:
        # "Kartımı ne zaman bloke edebilirim, politikanız nedir?" bir politika
        # sorusu, kart işlemi değil — eylem niyetleri soru niyetlerinden daha
        # güçlü kanıt istemeli, yoksa tool_agent alakasız bir "kartının son 4
        # hanesi?" sorusuyla RAG cevabını kirletiyor (bkz.
        # test_rule_based_extra_intents_requires_corroboration_for_action_intents).
        # Arızalı kart ifadeleri ise spesifik, o yüzden güvenle eklenebiliyor.
        "kart çalışm",
        "kartım çalışm",
        "kartim calism",
        "block my card",
        "card stolen",
        "lost my card",
    ),
    IntentLabel.SMALL_TALK: (
        "merhaba",
        "selam",
        "teşekkür",
        "sağol",
        "nasılsın",
        "nasıl gidiyor",
        "naber",
        "günaydın",
        "iyi günler",
        "iyi akşam",
        "hello",
        "thanks",
        "thank you",
        "how are you",
    ),
    # İnsan isteme kalıplarının yanında, para gönderme gibi diğer üç işlem
    # etiketinin kapsamadığı talepler de burada (bkz. TRANSACTION_ACTION notu).
    IntentLabel.ESCALATE: (
        "temsilciyle görüş",
        "temsilciye bağlan",
        "temsilciye aktar",
        # "temsilci" kökü: temsilcisine, temsilciyle, temsilcinize.
        "temsilci",
        "müşteri temsilcisi",
        # Şikayet dili eskalasyon sinyali: kızgın kullanıcıyı otomatik akışta
        # tutmak, memnuniyetsizliği büyütmekten başka bir şey yapmıyor.
        "şikayet",
        "şikayetçi",
        "dava",
        "müşteri temsilcisine",
        "müşteri hizmetleri",
        "insana bağla",
        "insanla görüş",
        "insanla konuş",
        "gerçek bir kişi",
        "gerçek bir insan",
        "gerçek biri",
        "yetkili biri",
        "yetkiliyle görüş",
        "canlı destek",
        "bot değil",
        "biriyle görüş",
        "biriyle konuş",
        "çalışanınızla konuş",
        "speak to a human",
        "speak with a human",
        "representative",
        "human agent",
        "customer service",
        "real person",
        "hesap açtır",
        "hesap açmak istiyorum",
        "kredi başvurusu",
        "kredi çekmek istiyorum",
        "para transfer",
        "para gönder",
        "havale gönder",
        "havale yap",
        "eft yap",
        "eft gönder",
        "transfer yap",
        "send money",
        "wire transfer",
    ),
    # IntentLabel.OUT_OF_SCOPE bilinçli olarak bir anahtar kelime listesine
    # sahip değil — diğer her intent sıfır puan aldığında düşülen fallback,
    # eşleşmeye çalışılacak bir şey değil.
}

# Destekleyici bir entity tipi, anahtar kelime kanıtının üzerine mütevazı,
# katkı sağlayan bir sinyal — onun yerine geçmiyor — bu yüzden bir çarpan ya
# da baskın bir terim değil, sadece bir düz puan ekliyor.
_ENTITY_BOOSTS: dict[IntentLabel, tuple[EntityType, ...]] = {
    IntentLabel.CARD_ACTION: (EntityType.CARD_LAST4,),
    IntentLabel.ACCOUNT_ACTION: (EntityType.IBAN, EntityType.ACCOUNT_TYPE),
    # AMOUNT bilinçli olarak yok: bir tutar genelde "gönder" işaret ediyor, "göster" değil.
    IntentLabel.TRANSACTION_ACTION: (EntityType.IBAN,),
}

_ENTITY_BOOST_WEIGHT = 1
_CONFIDENCE_BASE = 0.4
_CONFIDENCE_PER_POINT = 0.15
_CONFIDENCE_CAP = 0.95  # a rule-based classifier should never claim full certainty
_OUT_OF_SCOPE_CONFIDENCE = 0.3


class _IntentClassification(BaseModel):
    """`llm.with_structured_output` için dahili parse hedefi — genel bir DTO değil."""

    intent: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    extra_intents: list[IntentLabel] = Field(default_factory=list)


# Anahtar kelimeler doğru Türkçe ile yazılıyor; eşleştirme ise ASCII'ye
# indirgenmiş metin üzerinde yapılıyor. Bu yüzden listeyi bir kez indirgeyip
# saklıyoruz — her çağrıda yeniden katlamak, sıcak yoldaki tek gereksiz iş
# olurdu.
_FOLDED_INTENT_KEYWORDS: dict[IntentLabel, tuple[str, ...]] = {
    intent: tuple(ascii_fold(keyword) for keyword in keywords)
    for intent, keywords in _INTENT_KEYWORDS.items()
}


def _score_intents(text: str, entities: list[Entity]) -> dict[IntentLabel, int]:
    """Niyet skoru — ASCII'ye indirgenmiş metin üzerinde.

    Türkçe karakter kullanmayan kullanıcı ("musteri", "gormek", "hesabimda")
    aksi halde hiçbir anahtar kelimeyi tutturamıyor ve OUT_OF_SCOPE'a düşüyor.
    Gerçek kullanıcı dili setinde bu tek sınıf, hataların yarısından fazlasıydı.
    """
    folded = ascii_fold(text)
    entity_types = {entity.type for entity in entities}

    scores: dict[IntentLabel, int] = {
        intent: sum(1 for keyword in keywords if keyword in folded)
        for intent, keywords in _FOLDED_INTENT_KEYWORDS.items()
    }
    for intent, boost_types in _ENTITY_BOOSTS.items():
        if any(entity_type in entity_types for entity_type in boost_types):
            scores[intent] = scores.get(intent, 0) + _ENTITY_BOOST_WEIGHT
    return scores


def classify_intent_rule_based(text: str, entities: list[Entity]) -> tuple[IntentLabel, float]:
    scores = _score_intents(text, entities)
    best_intent = max(scores, key=lambda intent: scores[intent])
    best_score = scores[best_intent]
    if best_score <= 0:
        return IntentLabel.OUT_OF_SCOPE, _OUT_OF_SCOPE_CONFIDENCE

    confidence = min(_CONFIDENCE_CAP, _CONFIDENCE_BASE + _CONFIDENCE_PER_POINT * best_score)
    return best_intent, confidence


_EXTRA_INTENT_MIN_SCORE = 1

# CARD_ACTION/ACCOUNT_ACTION/TRANSACTION_ACTION tetiklenince gerçek bir işlem
# çalışır (ADR-009) — extra intent için çok kelimeli, spesifik bir kalıp
# gerekiyor. Regresyon: paylaşılan bir entity boost tek bir IBAN'ı iki intent'e
# birden kanıt sayıp gereksiz bir ikinci aracı tetikliyordu.
# RAG_QUERY/SMALL_TALK düşük riskli, tek eşleşmeyle yetiniyor.
_EXTRA_INTENT_CORROBORATION_REQUIRED = frozenset(_ENTITY_BOOSTS.keys())


def _has_strong_extra_intent_signal(intent: IntentLabel, folded_text: str) -> bool:
    return any(
        " " in keyword and keyword in folded_text
        for keyword in _FOLDED_INTENT_KEYWORDS.get(intent, ())
    )


def _rule_based_extra_intents(
    text: str, entities: list[Entity], primary: IntentLabel
) -> list[IntentLabel]:
    """Kural tabanlı yolun kendi çoklu-niyet tespiti — fake modda da ADR-012'nin
    tetiklenebilmesi için. `intent_agent._clean_extra_intents` aynı chainable-
    set/dedup/2-sınır filtresini gerçek LLM yolundakiyle aynı şekilde uyguluyor.
    """
    scores = _score_intents(text, entities)
    folded = ascii_fold(text)

    candidates = []
    for intent, score in scores.items():
        if intent == primary or score < _EXTRA_INTENT_MIN_SCORE:
            continue
        if intent in _EXTRA_INTENT_CORROBORATION_REQUIRED and not _has_strong_extra_intent_signal(
            intent, folded
        ):
            continue
        candidates.append(intent)
    return sorted(candidates, key=lambda intent: scores[intent], reverse=True)


async def classify_intent(
    text: str, entities: list[Entity], llm: BaseChatModel
) -> tuple[IntentLabel, float, list[IntentLabel]]:
    """Birincil niyeti ve varsa ek niyetleri döner.

    `extra_intents`, tek mesajda "kartımı blokla ve EFT limitiniz ne kadar"
    gibi birden fazla, farklı kategoriden isteği ayırt etmek için var (bkz.
    ADR-012). Gerçek modda bunu LLM'in structured output'u üretiyor; kural
    tabanlı yol da `_rule_based_extra_intents` ile kendi (daha kaba, keyword
    tabanlı) versiyonunu üretiyor — aksi halde bu özellik anahtarsız hiç
    görünmezdi.
    """
    if is_fake_model(llm):
        intent, confidence = classify_intent_rule_based(text, entities)
        extra_intents = _rule_based_extra_intents(text, entities, intent)
        return intent, confidence, extra_intents

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
        return result.intent, result.confidence, result.extra_intents
    except Exception:
        logger.warning("intent_llm_classification_failed", text_preview=text[:120], exc_info=True)
        intent, confidence = classify_intent_rule_based(text, entities)
        extra_intents = _rule_based_extra_intents(text, entities, intent)
        return intent, confidence, extra_intents
