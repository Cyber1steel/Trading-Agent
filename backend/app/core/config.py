from functools import lru_cache
from pathlib import Path

from pydantic import Field
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

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
