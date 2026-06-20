from __future__ import annotations

import logging

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    groq_api_key: str = Field(
        ...,
        validation_alias=AliasChoices("groq_api_key", "grok_api_key"),
    )
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1",
        validation_alias=AliasChoices("groq_base_url", "grok_base_url"),
    )
    model_a: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices("model_a"),
    )
    model_b: str = Field(
        default="openai/gpt-oss-120b",
        validation_alias=AliasChoices("model_b"),
    )
    judge_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        validation_alias=AliasChoices(
            "judge_model_name", "groq_model_name", "grok_model_name", "model_name"
        ),
    )
    max_retries: int = Field(default=5)
    retry_backoff_seconds: float = Field(default=2.0)
    request_timeout_seconds: float = Field(default=30.0)
    judge_samples: int = Field(
        default=3,
        description="Number of LLM judge calls for tone fidelity multi-sampling",
    )


def get_settings() -> Settings:
    """Return a fully-initialised Settings object."""
    settings = Settings()  # type: ignore[call-arg]
    logger.info(
        "Config loaded | model_a=%s model_b=%s judge_model=%s base_url=%s",
        settings.model_a,
        settings.model_b,
        settings.judge_model_name,
        settings.groq_base_url,
    )
    return settings
