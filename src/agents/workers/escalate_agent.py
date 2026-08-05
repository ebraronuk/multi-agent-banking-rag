"""Handles IntentLabel.ESCALATE and IntentLabel.OUT_OF_SCOPE.

Deliberately LLM-free: a request that's out of scope or explicitly asking for
a human doesn't need a generated response, it needs a consistent, predictable
handoff message. Generating one from an LLM here would just add latency and a
place for the model to improvise a promise the bank can't keep.
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
