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
