"""Ortam yapılandırması için tek erişim noktası.

Kod tabanındaki hiçbir yer `os.environ`'a doğrudan bakmamalı — `get_settings()` import edilir.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    FAKE = "fake"


class EmbeddingProvider(StrEnum):
    OPENAI = "openai"
    FAKE = "fake"


class AppEnv(StrEnum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: AppEnv = AppEnv.LOCAL
    log_level: str = "INFO"

    llm_provider: LLMProvider = LLMProvider.FAKE
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"

    embedding_provider: EmbeddingProvider = EmbeddingProvider.FAKE
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_persist_dir: str = "./data/vectorstore"
    chroma_collection: str = "banking_kb"

    mcp_server_host: str = "127.0.0.1"
    mcp_server_port: int = 8765
    force_in_process_tools: bool = Field(
        default=False,
        description="True ise gerçek bir LLM sağlayıcısıyla bile in-process araç "
        "istemcisi kullanılır (ayrı bir MCP sunucu süreci olmayan tek-konteynerli "
        "dağıtımlar için — bkz. agents/tools/mcp_client.py::get_tool_client).",
    )

    max_agent_iterations: int = Field(default=6, ge=1, le=20)

    # --- Oturum açmış müşteri (demo) ------------------------------------------
    # Gerçek üründe bu değerler oturum belirtecinden gelir. Burada sabit:
    # amaç "asistan kullanıcıyı tanır" varsayımını sisteme sokmak, gerçek bir
    # kimlik doğrulama kurmak değil (bkz. app/core/session.py).
    demo_customer_name: str = "Ayşe Demir"
    demo_account_id: str = "TR330006100519786457841326"

    # --- Gözlemlenebilirlik (Langfuse) ---------------------------------------
    # LLM sistemlerinde asıl zorluk hatayı görmek değil, hangi katmanda
    # olduğunu görmek. Trace'siz bir ajan grafiğinde "yanlış cevap"ın
    # retrieval'dan mı, prompt'tan mı, modelden mi geldiği tahmine kalıyor.
    # Anahtar verilmezse sessizce kapanır — `llm_provider=fake` ile aynı
    # felsefe: eksik yapılandırma çalışmayı durdurmamalı.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = Field(
        default=True,
        description="False ise anahtarlar dolu olsa bile tracing kapatılır "
        "(CI ve offline çalıştırmalar için açık kapı).",
    )
    langfuse_sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace'lenecek isteklerin oranı. Üretimde maliyet ve "
        "gürültüyü düşürmek için 1.0'ın altına çekilebilir.",
    )
    request_timeout_seconds: int = Field(default=30, ge=1)
    chat_rate_limit: str = Field(
        default="20/minute", description="slowapi limit string, e.g. '20/minute'"
    )

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None

    redis_url: str | None = Field(
        default=None,
        description="Boşsa konuşma hafızası bellek-içi bir dict'e düşer "
        "(lokal geliştirme/testler için sorun değil, restart'ta kaybolur, replikalar arası paylaşılmaz).",
    )
    database_url: str | None = Field(
        default=None,
        description="Bankacılık verisi (accounts/cards/transactions) için Postgres DSN'i. "
        "Boşsa bankacılık araçları bellek-içi bir dict'e düşer (bkz. db/schema.sql).",
    )
    conversation_history_limit: int = Field(default=6, ge=1, le=50)
    conversation_ttl_seconds: int = Field(default=86400, ge=60)

    cors_allowed_origins: str = Field(
        default="http://localhost:3000",
        description="Comma-separated list of origins the frontend is served from.",
    )

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]

    @property
    def mcp_server_url(self) -> str:
        # "/mcp" FastMCP'nin varsayılan streamable-HTTP yolu.
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}/mcp"

    @property
    def tracing_enabled(self) -> bool:
        """Tracing yalnızca açıkça açıldıysa VE iki anahtar da varsa aktif.

        Eksik anahtarla çökmek yerine kapanıyor: gözlemlenebilirlik bir
        teşhis aracı, bir çalışma önkoşulu değil.
        """
        return bool(self.langfuse_enabled and self.langfuse_public_key and self.langfuse_secret_key)

    def resolved_llm_provider(self) -> LLMProvider:
        """Anahtar yoksa çökmek yerine FAKE'e düşer — CI ve anahtarsız lokal
        çalıştırma her zaman ayakta kalmalı, sessizce özelliği kapatmak yerine
        deterministik sahte istemci devreye girer."""
        if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            return LLMProvider.FAKE
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            return LLMProvider.FAKE
        if self.llm_provider == LLMProvider.GOOGLE and not self.google_api_key:
            return LLMProvider.FAKE
        return self.llm_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
