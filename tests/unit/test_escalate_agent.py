from __future__ import annotations

from agents.state import new_state
from agents.workers.escalate_agent import escalate_node
from schemas.dto import IntentLabel


def test_escalate_intent_gets_human_handoff_message() -> None:
    state = new_state("c1", "bir insanla konuşmak istiyorum")
    state["intent"] = IntentLabel.ESCALATE

    result = escalate_node(state)

    assert "temsilci" in result["draft_answer"].lower()


def test_out_of_scope_gets_scope_message() -> None:
    state = new_state("c1", "yarın hava nasıl olacak")
    state["intent"] = IntentLabel.OUT_OF_SCOPE

    result = escalate_node(state)

    assert "kapsamı dışında" in result["draft_answer"].lower()
