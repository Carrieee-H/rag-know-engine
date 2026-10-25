from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # PostgreSQL
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "know_engine"
    pg_password: str = "know_engine"
    pg_db: str = "know_engine"

    # Elasticsearch
    es_url: str = "http://localhost:9200"
    es_index_alias: str = "know-engine-chunks"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "know-engine"
    minio_secure: bool = False

    # MinerU
    mineru_url: str = "http://localhost:8000"

    # Model API (OpenAI compatible)
    llm_base_url: str = ""
    llm_api_key: str = ""
    chat_model: str = "qwen3.8:27b"
    light_model: str = "qwen3.5:9b"
    vision_model: str = ""  # empty disables image descriptions
    llm_reasoning_effort: str | None = "none"  # disables "thinking"; empty = model default
    embedding_model: str = "bge-m3"
    embedding_dims: int = 1024

    # Local reranker
    rerank_model: str = "BAAI/bge-reranker-base"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
