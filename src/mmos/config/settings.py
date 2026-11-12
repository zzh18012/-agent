from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderName(StrEnum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    """Process-wide configuration, read from MMOS_* environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="MMOS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProviderName = LLMProviderName.DEEPSEEK
    llm_model: str = "deepseek-chat"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    projects_dir: Path = Path("projects")
    logs_dir: Path = Path("logs")
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
