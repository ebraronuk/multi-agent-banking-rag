"""IntentLabel.SMALL_TALK'ı işler — selamlama, teşekkür, sohbet.

RAG ajanına gömülmek yerine kendi düğümü olarak duruyor ki bağlamında hiçbir
zaman retrieval ya da araç olmasın: bir sohbet turn'ünün bir politika
dokümanına alıntı yapmaya ya da `get_balance` çağırmaya işi yok.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agents.memory import history_to_messages
from agents.prompts.smalltalk_prompt import SMALLTALK_SYSTEM_PROMPT
from agents.state import GraphState
from app.core.llm import is_fake_model, safe_ainvoke
from nlp.intent_classifier import is_capability_question
from nlp.text_utils import ascii_fold
from schemas.dto import AgentTraceStep

_CANCELLED_MESSAGE = (
    "Tamam, o işlemi iptal ettim. Başka bir konuda yardımcı olabilir miyim?"
)
# Teşekküre karşılama mesajıyla cevap vermek robotik duruyordu: kullanıcı
# zaten tanışmış, kendini tekrar tanıtan bir asistan konuşmayı başa sarıyor.
_THANKS_MESSAGE = "Rica ederim. Başka bir konuda yardımcı olabilir miyim?"
_THANKS_MARKERS = ("teşekkür", "tesekkur", "sağol", "sagol", "eyvallah", "thanks", "mersi")


def _is_thanks(text: str) -> bool:
    lowered = text.strip().lower().replace("İ", "i")
    return len(lowered) <= 30 and any(marker in lowered for marker in _THANKS_MARKERS)


_FALLBACK_GREETING = (
    "Merhaba, ben DemoBank asistanıyım. Bakiye, işlem geçmişi, kart işlemleri "
    "ve banka hizmetleriyle ilgili sorularınızda yardımcı olabilirim."
)
# Konuşma zaten başlamışsa kendini yeniden tanıtmak robotik duruyor: aynı
# tanıtım cümlesini ikinci kez duymak, karşıdakinin konuşmayı hatırlamadığı
# izlenimi veriyor.
_RETURNING_GREETING = "Buyurun, nasıl yardımcı olabilirim?"

# "Nasılsın" bir bankacılık sorusu değil ama cevapsız da bırakılamaz.
# Önceki davranış tam tanıtımı tekrar okuyordu — soruyu duymamış gibi.
_HOW_ARE_YOU_MESSAGE = "İyiyim, teşekkür ederim. Sizin için ne yapabilirim?"
_HOW_ARE_YOU_MARKERS = ("nasilsin", "naber", "nasil gidiyor", "how are you", "ne haber")

# Yetenek cevabı gerçek modelde de deterministik. Sebep: bu cümle ürünün
# kapsamını söylüyor. Bir modelden doğaçlamasını istemek, olmayan bir
# yeteneği ("para transferi yapabilirim") sayması riskini almak demek —
# kapsam beyanı, cevabın en az tahmine açık olması gereken yeri.
_CAPABILITY_MESSAGE = (
    "Ben DemoBank'ın dijital asistanıyım. Şunlarda yardımcı olabilirim:\n"
    "- Hesap bakiyenizi söyleyebilirim\n"
    "- Son işlemlerinizi listeleyebilirim\n"
    "- Kayıp ya da çalıntı kartınızı bloke edebilirim\n"
    "- Havale/EFT limitleri, hesap ücretleri gibi banka politikalarını yanıtlayabilirim\n"
    "Bunların dışında bir konu olursa sizi bir müşteri temsilcisine aktarabilirim."
)


def _is_how_are_you(text: str) -> bool:
    folded = ascii_fold(text)
    return len(folded) <= 30 and any(marker in folded for marker in _HOW_ARE_YOU_MARKERS)


def build_smalltalk_node(
    llm: BaseChatModel,
) -> Callable[[GraphState], Awaitable[dict[str, object]]]:
    async def smalltalk_node(state: GraphState) -> dict[str, object]:
        history = state.get("history", [])
        if state.get("cancelled_pending"):
            draft_answer = _CANCELLED_MESSAGE
        elif _is_thanks(state["user_query"]):
            draft_answer = _THANKS_MESSAGE
        elif is_capability_question(state["user_query"]):
            draft_answer = _CAPABILITY_MESSAGE
        elif _is_how_are_you(state["user_query"]):
            draft_answer = _HOW_ARE_YOU_MESSAGE
        elif is_fake_model(llm):
            draft_answer = _RETURNING_GREETING if history else _FALLBACK_GREETING
        else:
            draft_answer = await safe_ainvoke(
                llm,
                [
                    SystemMessage(content=SMALLTALK_SYSTEM_PROMPT),
                    *history_to_messages(history),
                    HumanMessage(content=state["user_query"]),
                ],
                node="smalltalk",
            ) or _FALLBACK_GREETING
        return {
            "draft_answer": draft_answer,
            "worker_pass_done": True,
            "trace": [AgentTraceStep(node="smalltalk", summary="generated conversational reply")],
        }

    return smalltalk_node
