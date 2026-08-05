"""Chat-model factory.

Worker'lar `langchain_core.language_models.BaseChatModel`'e bağımlı, hiçbir
zaman somut bir sağlayıcı sınıfına değil. Bu, `LLM_PROVIDER=fake`'i CI ve
offline geliştirme için doğrudan bir yer değiştirme yapıyor, Anthropic/OpenAI/
Google arasında geçişi de bir refactor yerine tek satırlık bir değişikliğe
indiriyor.
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
    """LLM_PROVIDER=fake için offline yerine geçen deterministik model (CI, testler, anahtarsız).

    Gerçek bir modele çıkmak yerine son insan mesajının hash'ini alıyor, yani
    aynı girdi her zaman aynı çıktıyı üretiyor — API kredisi harcamadan ya da
    ağa çıkmadan graf davranışını testlerde snapshot'lamak için kullanışlı.
    Araç-çağırma düğümleri bu modeli `bind_tools`'a güvenmek yerine doğrudan
    özel olarak ele alıyor (bkz. `agents/workers/tool_agent.py`) — bir hash'in
    "burada hangi araç geçerli" diye bir fikri yok çünkü.
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
        # Tirelerle ayrıldı ki bir hex digest hiçbir zaman 6+ haneli bir dizi
        # içermesin: hex karakterleri 0-9a-f olduğu için düz 8 karakterlik bir
        # digest kazara bir telefon/kart numarası parçasına benzeyebilir ve
        # guardrail'in PII redaksiyonunu (agents/workers/guardrail_agent.py)
        # tamamen zararsız bir sahte çıktı üzerinde tetikleyebilir.
        digest = "-".join(raw_digest[i : i + 2] for i in range(0, len(raw_digest), 2))
        content = f"{self.response_prefix} response-{digest}: {str(last_human)[:160]}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    provider = settings.resolved_llm_provider()

    if provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        # `resolved_llm_provider()` sadece anahtar set edilmişse ANTHROPIC
        # döner (bkz. Settings.resolved_llm_provider), o yüzden bu runtime'da
        # garanti olarak None değil — assert sadece mypy için tipi daraltıyor.
        assert settings.anthropic_api_key is not None
        return ChatAnthropic(
            model_name=settings.llm_model,
            api_key=SecretStr(settings.anthropic_api_key),
            timeout=settings.request_timeout_seconds,
            max_retries=2,
            stop=None,  # langchain-anthropic'in tipli __init__'i bunu açıkça istiyor, runtime'da zaten None
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
    """LLM'i çağırır, ham `AIMessage`'ı ya da çağrı patladıysa None döner.

    Düz bir `BaseChatModel` ya da bağlı bir `Runnable` (`llm.bind_tools(...)`'un
    döndürdüğü — tool_agent'ın akıl yürütme döngüsü bunu geçiyor) kabul eder,
    ikisi de aynı `.ainvoke()` sözleşmesini paylaşıyor. Tam mesajı (sadece
    `.content`'i değil) döndürmesinin sebebi, o döngünün `.tool_calls`'a da
    ihtiyaç duyması, sadece metne değil.
    """
    try:
        response = await llm.ainvoke(messages)
        return response if isinstance(response, AIMessage) else AIMessage(content=str(response.content))
    except Exception:
        logger.warning("llm_call_failed", node=node, exc_info=True)
        return None


async def safe_ainvoke(llm: BaseChatModel, messages: list[BaseMessage], *, node: str) -> str | None:
    """LLM'i çağırır, metnini ya da çağrı patladıysa None döner.

    `rag_agent`/`smalltalk_agent`/`tool_agent`'ın deterministik yolu, üretim
    çağrısını kendi try/except'ini üç kopya olarak yazmak yerine buradan
    geçiriyor: gerçek bir Anthropic/OpenAI/Google kesintisi, timeout ya da
    rate limit bu turn'ün cevabını bozmalı (set edilmemiş bir `draft_answer`
    zaten `guardrail_agent.py`'nin NO_DRAFT_PRODUCED yoluna akıyor), tüm
    `/chat` isteğini 500'e düşürmemeli.
    """
    response = await safe_ainvoke_message(llm, messages, node=node)
    return str(response.content) if response is not None else None
