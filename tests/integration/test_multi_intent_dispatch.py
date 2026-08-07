"""Gerçek derlenmiş grafiğe karşı: tek mesajda iki farklı niyetin (RAG_QUERY +
SMALL_TALK) ikisinin de işlenip tek bir cevapta birleştiğini uçtan uca doğrular
(bkz. ADR-012). Sahte modelde bu yol hiç tetiklenmediği için (extra_intents her
zaman boş), `get_chat_model`'i senaryoyu canlandıran bir script'li modelle,
`get_tool_client`'ı da ağa çıkmayan in-process istemciyle değiştiriyoruz —
`agents/graph.py` içindeki gerçek düğüm/kenar bağlantısı test ediliyor, bir
kopyası değil.
"""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agents.graph import build_graph
from agents.state import new_state
from agents.tools.mcp_client import InProcessToolClient
from app.core.config import Settings
from mcp_server.tools.banking_repository import InMemoryBankingRepository
from nlp.intent_classifier import _IntentClassification
from nlp.ner_extractor import _NERExtraction


class _FixedStructuredResult:
    """`with_structured_output(...).ainvoke(...)` sözleşmesini karşılayan sabit bir sonuç."""

    def __init__(self, value: object) -> None:
        self._value = value

    async def ainvoke(self, messages: object) -> object:
        return self._value


class _ScriptedMultiIntentModel(BaseChatModel):
    """`ner_agent`/`intent_agent`in structured-output çağrılarını ve
    `rag_agent`/`smalltalk`/`synthesizer`in düz `ainvoke` çağrılarını tek bir
    modelle karşılar — gerçek grafiğin her düğümünü olduğu gibi çalıştırmak
    için gereken minimum script.
    """

    intent_result: object = None
    ner_result: object = None
    plain_responses: list[AIMessage] = Field(default_factory=list)
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-multi-intent-test-model"

    def with_structured_output(self, schema: object, **kwargs: object) -> object:
        if schema is _IntentClassification:
            return _FixedStructuredResult(self.intent_result)
        if schema is _NERExtraction:
            return _FixedStructuredResult(self.ner_result)
        raise AssertionError(f"beklenmeyen structured-output şeması: {schema!r}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        index = min(self.call_index, len(self.plain_responses) - 1)
        self.call_index += 1
        return ChatResult(generations=[ChatGeneration(message=self.plain_responses[index])])


async def test_multi_intent_message_dispatches_to_both_workers_and_synthesizes() -> None:
    scripted = _ScriptedMultiIntentModel(
        intent_result=_IntentClassification(
            intent="RAG_QUERY", confidence=0.9, extra_intents=["SMALL_TALK"]
        ),
        ner_result=_NERExtraction(entities=[]),
        plain_responses=[
            AIMessage(content="EFT limitiniz günlük 50.000 TL'dir."),
            AIMessage(content="Merhaba, ben de iyiyim!"),
            AIMessage(content="EFT limitiniz 50.000 TL, ayrıca merhaba, ben de iyiyim!"),
        ],
    )
    settings = Settings(llm_provider="anthropic", anthropic_api_key="test-key")

    with (
        patch("agents.graph.get_chat_model", return_value=scripted),
        patch(
            "agents.graph.get_tool_client",
            return_value=InProcessToolClient(InMemoryBankingRepository()),
        ),
    ):
        graph = build_graph(settings)
        final_state = await graph.ainvoke(new_state("c1", "EFT limitiniz ne kadar, bu arada merhaba"))

    assert final_state["final_answer"] == "EFT limitiniz 50.000 TL, ayrıca merhaba, ben de iyiyim!"

    node_sequence = [step.node for step in final_state["trace"]]
    assert "rag_agent" in node_sequence
    assert "smalltalk" in node_sequence
    assert "advance_intent" in node_sequence
    assert "synthesizer" in node_sequence
    assert node_sequence.count("rag_agent") == 1
    assert node_sequence.count("smalltalk") == 1


async def test_single_intent_message_never_touches_advance_or_synthesizer() -> None:
    # Aynı script'li model, ama extra_intents boş — bu turda hiç ek niyet yok,
    # advance_intent/synthesizer düğümleri hiç çalışmamalı.
    scripted = _ScriptedMultiIntentModel(
        intent_result=_IntentClassification(intent="SMALL_TALK", confidence=0.95, extra_intents=[]),
        ner_result=_NERExtraction(entities=[]),
        plain_responses=[AIMessage(content="Merhaba, size nasıl yardımcı olabilirim?")],
    )
    settings = Settings(llm_provider="anthropic", anthropic_api_key="test-key")

    with (
        patch("agents.graph.get_chat_model", return_value=scripted),
        patch(
            "agents.graph.get_tool_client",
            return_value=InProcessToolClient(InMemoryBankingRepository()),
        ),
    ):
        graph = build_graph(settings)
        final_state = await graph.ainvoke(new_state("c1", "merhaba"))

    node_sequence = [step.node for step in final_state["trace"]]
    assert "advance_intent" not in node_sequence
    assert "synthesizer" not in node_sequence
    assert final_state["final_answer"] == "Merhaba, size nasıl yardımcı olabilirim?"
