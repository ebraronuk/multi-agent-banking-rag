"""Chat-model factory.

Workers depend on `langchain_core.language_models.BaseChatModel`, never on a
concrete provider class. That keeps `LLM_PROVIDER=fake` a drop-in swap for CI
and offline dev, and makes swapping Anthropic/OpenAI a one-line change instead
of a refactor.
"""

from __future__ import annotations

import hashlib

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import SecretStr

from app.core.config import LLMProvider, Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class FakeChatModel(BaseChatModel):
    """Deterministic offline stand-in for LLM_PROVIDER=fake (CI, tests, no API key).

    Hashes the last human message instead of calling out to a real model, so
    the same input always produces the same output — useful for snapshotting
    graph behaviour in tests without burning API credits or hitting the network.
    Tool-calling nodes special-case this model directly (see
    `agents/workers/tool_agent.py`) rather than relying on `bind_tools`, since a
    hash has no notion of "which tool applies here".
    """

    response_prefix: str = "[fake-llm]"

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        last_human = next((m.content for m in reversed(messages) if m.type == "human"), "")
        raw_digest = hashlib.sha256(str(last_human).encode()).hexdigest()[:8]
        # Hyphenated so a hex digest never contains a 6+ digit run: hex chars
        # are 0-9a-f, so a plain 8-char digest can accidentally look like a
        # phone number/PAN fragment and trip the guardrail's PII redaction
        # (agents/workers/guardrail_agent.py) on completely harmless fake output.
        digest = "-".join(raw_digest[i : i + 2] for i in range(0, len(raw_digest), 2))
        content = f"{self.response_prefix} response-{digest}: {str(last_human)[:160]}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    provider = settings.resolved_llm_provider()

    if provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        # `resolved_llm_provider()` only returns ANTHROPIC when the key is set
        # (see Settings.resolved_llm_provider), so this is guaranteed non-None
        # at runtime — the assert exists to narrow the type for mypy.
        assert settings.anthropic_api_key is not None
        return ChatAnthropic(
            model_name=settings.llm_model,
            api_key=SecretStr(settings.anthropic_api_key),
            timeout=settings.request_timeout_seconds,
            max_retries=2,
            stop=None,  # langchain-anthropic's typed __init__ requires this explicitly, though it defaults to None at runtime
        )

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI

        assert settings.openai_api_key is not None
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=SecretStr(settings.openai_api_key),
            timeout=settings.request_timeout_seconds,
            max_retries=2,
        )

    if provider == LLMProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI

        assert settings.google_api_key is not None
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=SecretStr(settings.google_api_key),
            timeout=settings.request_timeout_seconds,
            max_retries=2,
        )

    return FakeChatModel()


def is_fake_model(model: BaseChatModel) -> bool:
    return isinstance(model, FakeChatModel)


async def safe_ainvoke_message(
    llm: BaseChatModel | Runnable[list[BaseMessage], AIMessage],
    messages: list[BaseMessage],
    *,
    node: str,
) -> AIMessage | None:
    """Call the LLM and return the raw `AIMessage`, or None if the call raised.

    Accepts a plain `BaseChatModel` or a bound `Runnable` (what
    `llm.bind_tools(...)` returns — `tool_agent`'s reasoning loop passes one
    of those) — both share the same `.ainvoke()` contract. Returning the full
    message (not just `.content`) is what that loop needs: it has to read
    `.tool_calls`, not only text.
    """
    try:
        response = await llm.ainvoke(messages)
        return response if isinstance(response, AIMessage) else AIMessage(content=str(response.content))
    except Exception:
        logger.warning("llm_call_failed", node=node, exc_info=True)
        return None


async def safe_ainvoke(llm: BaseChatModel, messages: list[BaseMessage], *, node: str) -> str | None:
    """Call the LLM and return its text, or None if the call raised.

    `rag_agent`/`smalltalk_agent`/`tool_agent`'s deterministic path all funnel
    their generation call through here rather than each catching provider
    exceptions themselves: a real Anthropic/OpenAI outage, rate limit, or
    timeout should degrade this turn's answer (an unset `draft_answer`
    already flows into `guardrail_agent.py`'s NO_DRAFT_PRODUCED fallback),
    not 500 the whole `/chat` request. One hardened call site instead of
    three copies of the same try/except.
    """
    response = await safe_ainvoke_message(llm, messages, node=node)
    return str(response.content) if response is not None else None
