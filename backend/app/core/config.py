from functools import lru_cache
from pathlib import Path
import re
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Trading Agent API"
    app_env: str = "development"
    database_url: str = "postgresql+psycopg://localhost/trading_agent"
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = Field(default=384, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_cache_dir: Path = Path(".cache/fastembed")
    embedding_local_files_only: bool = True
    hybrid_semantic_candidates: int = Field(default=20, gt=0)
    hybrid_lexical_candidates: int = Field(default=20, gt=0)
    hybrid_rrf_k: int = Field(default=60, gt=0)
    reasoning_provider_order: str = "groq,openai"
    reasoning_model: str = "gpt-6.1-sol"
    groq_model: str = "openai/gpt-oss-120b"
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    reasoning_openai_base_url: str | None = None
    reasoning_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    reasoning_max_output_tokens: int = Field(default=1600, ge=128, le=8192)
    reasoning_max_input_bytes: int = Field(default=65536, ge=1024, le=1048576)
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"

    @field_validator("reasoning_provider_order", mode="before")
    @classmethod
    def normalize_provider_order(cls, value):
        if not isinstance(value, str):
            raise ValueError("reasoning_provider_order must be a comma-separated string")
        providers = tuple(part.strip().lower() for part in value.split(","))
        if (
            not providers or any(not provider for provider in providers)
            or len(providers) > 2 or len(set(providers)) != len(providers)
            or any(provider not in {"groq", "openai"} for provider in providers)
        ):
            raise ValueError("reasoning_provider_order must list one or two unique supported providers")
        return ",".join(providers)

    @field_validator("groq_model", "reasoning_model")
    @classmethod
    def nonblank_model_id(cls, value):
        if (
            not isinstance(value, str) or not value.strip() or len(value) > 200
            or re.fullmatch(r"[A-Za-z0-9._/-]+", value.strip()) is None
            or value.strip().lower().startswith(("gsk_", "sk-"))
        ):
            raise ValueError("reasoning model identifier must be a non-empty string")
        return value.strip()

    @field_validator("reasoning_openai_base_url", mode="before")
    @classmethod
    def normalize_reasoning_base_url(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if not isinstance(value, str):
            raise ValueError("reasoning_openai_base_url must be a URL string")
        parsed = urlsplit(value.strip())
        local_http = parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and not local_http:
            raise ValueError("reasoning provider base URL must use HTTPS (HTTP is allowed only for localhost)")
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("reasoning provider base URL must not contain credentials, query, or fragment")
        return value.strip().rstrip("/")

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
