"""IntentLabel.ESCALATE ve IntentLabel.OUT_OF_SCOPE'u işler.

Bilinçli olarak LLM'siz: kapsam dışı bir istek ya da açıkça insan isteyen bir
talep, üretilmiş bir cevaba değil, tutarlı ve öngörülebilir bir aktarım
mesajına ihtiyaç duyar. Burada bir modele ürettirmek sadece gecikme ve
bankanın tutamayacağı bir vaadi modelin doğaçlama etmesi için bir fırsat
eklerdi.
"""

from __future__ import annotations

from agents.state import GraphState
from schemas.dto import AgentTraceStep, IntentLabel

_ESCALATE_MESSAGE = (
    "Sizi bir müşteri temsilcisine aktarıyorum. Görüşme geçmişiniz temsilciyle paylaşılacak, "
    "tekrar baştan anlatmanıza gerek kalmayacak."
)
_OUT_OF_SCOPE_MESSAGE = (
    "Bu konu bankacılık asistanının kapsamı dışında; bu nedenle sağlıklı bir yanıt veremem. "
    "Bankacılık hizmetleriyle ilgili bir sorunuz varsa yardımcı olmaktan memnuniyet duyarım."
)


def escalate_node(state: GraphState) -> dict[str, object]:
    message = (
        _ESCALATE_MESSAGE if state.get("intent") == IntentLabel.ESCALATE else _OUT_OF_SCOPE_MESSAGE
    )
    return {
        "draft_answer": message,
        "trace": [
            AgentTraceStep(node="escalate", summary=f"handed off for intent={state.get('intent')}")
        ],
    }
