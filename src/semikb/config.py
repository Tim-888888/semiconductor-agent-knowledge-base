"""Application configuration loaded from the repository-root .env file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration with safe defaults for the synthetic demo mode."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    demo_mode: bool = True
    api_prefix: str = "/api/v1"
    jwt_secret: str = "change-this-demo-secret-before-shared-use"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    mongodb_uri: str = ""
    mongodb_database: str = "semikb"
    milvus_uri: str = ""
    milvus_token: str = ""
    minio_endpoint: str = ""
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    redis_url: str = ""
    milvus_index_version: str = "v2"
    milvus_require_active_alias: bool = True

    mineru_api_base_url: str = ""
    mineru_api_key: str = ""
    mineru_model_version: str = "vlm"
    mineru_timeout_seconds: int = Field(default=900, ge=30, le=3600)
    mineru_poll_seconds: int = Field(default=3, ge=1, le=30)
    llm_api_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    aliyun_web_mcp_url: str = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
    aliyun_web_mcp_api_key: str = ""
    aliyun_web_mcp_tool_name: str = "web_search"
    web_allowed_domains: str = ""

    bge_m3_model_path: str = ""
    bge_reranker_model_path: str = ""
    bge_use_fp16: bool = False
    embedding_dim: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)

    @property
    def allowed_domains(self) -> tuple[str, ...]:
        return tuple(domain.strip().lower() for domain in self.web_allowed_domains.split(",") if domain.strip())

    @property
    def external_storage_configured(self) -> bool:
        return bool(self.mongodb_uri and self.milvus_uri and self.minio_endpoint and self.redis_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
