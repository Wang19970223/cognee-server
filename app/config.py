"""
app/config.py
─────────────────────────────────────────────────────────────────────────────
All service configuration loaded from environment variables / .env file.
Must be imported BEFORE importing cognee so env vars are set in time.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Service ──────────────────────────────────────────────────────────────
    service_port: int = 8000
    # Comma-separated list of API keys with permission levels:
    #   COGNEE_API_KEYS=key1:read,key2:write,key3:admin
    cognee_api_keys: str = ""

    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM / Embedding (DashScope / OpenAI-compatible) ──────────────────────
    llm_api_key: str = ""
    llm_endpoint: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "openai/qwen-plus"
    embedding_model: str = "openai/text-embedding-v3"
    embedding_dimensions: str = "1024"

    # ── Cognee dataset defaults ───────────────────────────────────────────────
    cognee_dataset: str = "knowledge"

    # ── Network proxy (corporate network) ────────────────────────────────────
    http_proxy: str = ""
    https_proxy: str = ""

    # ── Cognee internals (forwarded to env) ──────────────────────────────────
    db_provider: str = "sqlite"
    db_name: str = "cognee_db"
    graph_database_provider: str = "kuzu"
    graph_dataset_database_handler: str = "kuzu"
    vector_db_provider: str = "lancedb"
    enable_backend_access_control: str = "false"
    # Data stored in project directory rather than inside .venv
    cognee_data_root_directory: str = ".cognee_data"

    def parsed_api_keys(self) -> Dict[str, str]:
        """Return {api_key: permission_level} from COGNEE_API_KEYS env var."""
        result: Dict[str, str] = {}
        for entry in self.cognee_api_keys.split(","):
            entry = entry.strip()
            if ":" in entry:
                key, level = entry.split(":", 1)
                result[key.strip()] = level.strip().lower()
            elif entry:
                result[entry] = "read"
        return result

    def apply_to_environ(self) -> None:
        """Inject all cognee-related settings into os.environ before cognee import."""
        os.environ.setdefault("LLM_PROVIDER", "openai")
        os.environ["LLM_MODEL"] = self.llm_model
        os.environ["LLM_ENDPOINT"] = self.llm_endpoint
        os.environ["LLM_API_KEY"] = self.llm_api_key
        os.environ["EMBEDDING_PROVIDER"] = "openai"
        os.environ["EMBEDDING_MODEL"] = self.embedding_model
        os.environ["EMBEDDING_ENDPOINT"] = self.llm_endpoint
        os.environ["EMBEDDING_API_KEY"] = self.llm_api_key
        os.environ.pop("EMBEDDING_DIMENSIONS", None)
        os.environ["EMBEDDING_DIMENSIONS"] = self.embedding_dimensions
        os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
        os.environ["TELEMETRY_DISABLED"] = "1"
        os.environ["LOG_LEVEL"] = "ERROR"
        os.environ["COGNEE_LOG_LEVEL"] = "ERROR"
        os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = self.enable_backend_access_control
        os.environ["DB_PROVIDER"] = self.db_provider
        os.environ["DB_NAME"] = self.db_name
        os.environ["GRAPH_DATABASE_PROVIDER"] = self.graph_database_provider
        os.environ["GRAPH_DATASET_DATABASE_HANDLER"] = self.graph_dataset_database_handler
        os.environ["VECTOR_DB_PROVIDER"] = self.vector_db_provider
        os.environ["COGNEE_DATA_ROOT_DIRECTORY"] = self.cognee_data_root_directory
        if self.http_proxy:
            os.environ["HTTP_PROXY"] = self.http_proxy
            os.environ["HTTPS_PROXY"] = self.http_proxy


@lru_cache
def get_settings() -> Settings:
    return Settings()
