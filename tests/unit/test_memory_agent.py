"""`agents/workers/memory_agent.py` için birim testler.

`InMemoryMemory` gerçek şeyin ta kendisi (mock'a gerek yok, bkz. test_memory.py),
o yüzden burada node'ların onun üzerinden doğru state anahtarlarını okuyup
yazdığını doğruluyoruz — özellikle `carried_escalation_stage`/`escalation_stage`
(bkz. agents/workers/escalate_agent.py, ADR-013).
"""

from __future__ import annotations

from agents.memory import InMemoryMemory
from agents.state import new_state
from agents.workers.memory_agent import build_memory_load_node, build_memory_save_node
from app.core.config import Settings


async def test_memory_load_exposes_carried_escalation_stage() -> None:
    memory = InMemoryMemory()
    await memory.save_turn(
        "conv-1",
        "insanla konuşmak istiyorum",
        "sizi aktarıyorum",
        None,
        history_limit=6,
        escalation_stage="handed_off",
    )
    load_node = build_memory_load_node(memory)

    result = await load_node(new_state("conv-1", "aktarım yapıldı mı"))

    assert result["carried_escalation_stage"] == "handed_off"
    assert "escalation stage=handed_off" in result["trace"][0].summary
    # `escalation_stage` (bu turun çıkışı) de aynı değerle varsayılan olarak
    # başlıyor — escalate_node'u atlayan bir turda (bkz. supervisor.py'deki
    # awaiting_issue+RAG_QUERY istisnası) aşama sessizce kaybolmasın diye.
    assert result["escalation_stage"] == "handed_off"


async def test_memory_load_defaults_carried_escalation_stage_none_for_new_conversation() -> None:
    memory = InMemoryMemory()
    load_node = build_memory_load_node(memory)

    result = await load_node(new_state("never-seen", "merhaba"))

    assert result["carried_escalation_stage"] is None


async def test_memory_save_persists_escalation_stage_from_state() -> None:
    memory = InMemoryMemory()
    save_node = build_memory_save_node(memory, Settings())
    state = new_state("conv-2", "bir insanla konuşmak istiyorum")
    state["final_answer"] = "Sizi bir müşteri temsilcisine aktarıyorum."
    state["escalation_stage"] = "handed_off"

    await save_node(state)
    context = await memory.load("conv-2")

    assert context.escalation_stage == "handed_off"


async def test_memory_save_defaults_escalation_stage_none_when_state_never_sets_it() -> None:
    memory = InMemoryMemory()
    save_node = build_memory_save_node(memory, Settings())
    state = new_state("conv-3", "bakiyem ne kadar")
    state["final_answer"] = "1000 TL"

    await save_node(state)
    context = await memory.load("conv-3")

    assert context.escalation_stage is None
