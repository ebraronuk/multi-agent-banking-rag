"""`agents/workers/synthesizer_agent.py` için birim testler.

Fake modelde birleştirme deterministik (taslaklar art arda eklenir); gerçek
modelde `safe_ainvoke` üzerinden bir çağrı yapılıyor, başarısız olursa aynı
deterministik birleştirmeye düşüyor (bkz. tool_agent/rag_agent'taki aynı desen).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.state import new_state
from agents.workers.synthesizer_agent import build_synthesizer_node
from app.core.llm import FakeChatModel


class _StubMergeModel:
    def __init__(self, response_text: str | None = None, raises: bool = False) -> None:
        self._response_text = response_text
        self._raises = raises

    async def ainvoke(self, messages: object) -> AIMessage:
        if self._raises:
            raise RuntimeError("provider outage")
        return AIMessage(content=self._response_text or "")


async def test_fake_model_concatenates_drafts_in_order() -> None:
    node = build_synthesizer_node(FakeChatModel())
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["collected_drafts"] = ["kartınız bloklandı"]
    state["draft_answer"] = "EFT limitiniz günlük 50.000 TL'dir."

    result = await node(state)

    assert result["draft_answer"] == "kartınız bloklandı\n\nEFT limitiniz günlük 50.000 TL'dir."
    assert "merged 2" in result["trace"][0].summary


async def test_real_model_merges_drafts_into_one_natural_answer() -> None:
    stub = _StubMergeModel(response_text="Kartınızı bloklandı, ayrıca EFT limitiniz 50.000 TL.")
    node = build_synthesizer_node(stub)  # type: ignore[arg-type]
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["collected_drafts"] = ["kartınız bloklandı"]
    state["draft_answer"] = "EFT limitiniz günlük 50.000 TL'dir."

    result = await node(state)

    assert result["draft_answer"] == "Kartınızı bloklandı, ayrıca EFT limitiniz 50.000 TL."


async def test_real_model_falls_back_to_concatenation_on_failure() -> None:
    stub = _StubMergeModel(raises=True)
    node = build_synthesizer_node(stub)  # type: ignore[arg-type]
    state = new_state("c1", "kartımı blokla ve EFT limitiniz ne kadar")
    state["collected_drafts"] = ["kartınız bloklandı"]
    state["draft_answer"] = "EFT limitiniz günlük 50.000 TL'dir."

    result = await node(state)

    assert result["draft_answer"] == "kartınız bloklandı\n\nEFT limitiniz günlük 50.000 TL'dir."


async def test_handles_two_collected_drafts_with_no_trailing_current_draft() -> None:
    node = build_synthesizer_node(FakeChatModel())
    state = new_state("c1", "x")
    state["collected_drafts"] = ["A", "B"]
    state["draft_answer"] = None

    result = await node(state)

    assert result["draft_answer"] == "A\n\nB"
