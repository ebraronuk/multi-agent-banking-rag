"""Single access point for environment configuration (Zod-equivalent: pydantic-settings).

Nothing else in the codebase should read `os.environ` directly — import `get_settings()`.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
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
    llm_model: str = "claude-sonnet-5"

    embedding_provider: EmbeddingProvider = EmbeddingProvider.FAKE
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_persist_dir: str = "./data/vectorstore"
    chroma_collection: str = "banking_kb"

    mcp_server_host: str = "127.0.0.1"
    mcp_server_port: int = 8765

    max_agent_iterations: int = Field(default=6, ge=1, le=20)
    request_timeout_seconds: int = Field(default=30, ge=1)
    chat_rate_limit: str = Field(
        default="20/minute", description="slowapi limit string, e.g. '20/minute'"
    )

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None

    @property
    def mcp_server_url(self) -> str:
        # "/mcp" is FastMCP's default streamable-HTTP path (`fastmcp.settings.streamable_http_path`)
        # — the path `mcp_server.server` actually serves on, verified against the pinned version.
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}/mcp"

    def resolved_llm_provider(self) -> LLMProvider:
        """Fail open to FAKE instead of crashing when no key is configured.

        A portfolio/demo deployment (and CI) must boot and serve traffic without
        real credentials; the fake client keeps behaviour deterministic instead
        of silently disabling the feature.
        """
        if self.llm_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            return LLMProvider.FAKE
        if self.llm_provider == LLMProvider.OPENAI and not self.openai_api_key:
            return LLMProvider.FAKE
        return self.llm_provider


@lru_cache
def get_settings() -> Settings:
    return Settings()
