"""Runtime settings, read from environment variables and the project's .env file.

Precedence (highest first): real environment variables, then .env, then the defaults below.
Secrets are typed as SecretStr so they never show up in logs or reprs.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Vision-language model
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    llm_max_attempts: int = Field(4, ge=1, le=10)
    llm_timeout_seconds: float = Field(120, gt=0)
    llm_max_concurrency: int = Field(2, ge=1, le=32)  # parallel API calls; keep low on the free tier
    llm_send_width_px: int = Field(1240, ge=600)

    # Pipeline
    render_dpi: int = Field(200, ge=100, le=600)
    max_workers: int = Field(4, ge=1, le=32)
    cache_dir: Path = Path("output/cache")
    log_level: str = "INFO"

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _empty_key_means_none(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @property
    def has_llm(self) -> bool:
        return self.gemini_api_key is not None


@lru_cache
def get_settings() -> Settings:
    return Settings()
