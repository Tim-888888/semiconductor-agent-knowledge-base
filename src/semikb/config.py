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
    milvus_index_version: str = "v3"
    milvus_require_active_alias: bool = True

    mineru_api_base_url: str = ""
    mineru_api_key: str = ""
    mineru_model_version: str = "vlm"
    mineru_timeout_seconds: int = Field(default=900, ge=30, le=3600)
    mineru_poll_seconds: int = Field(default=3, ge=1, le=30)
    llm_api_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_primary_provider: str = "closeai"
    llm_fallback_provider: str = "qwen"
    llm_timeout_seconds: int = Field(default=60, ge=5, le=600)
    closeai_base_url: str = ""
    closeai_api_key: str = ""
    closeai_model: str = "gpt-5.6-luna"
    closeai_reasoning_effort: str = "none"
    closeai_verbosity: str = "low"
    qwen_api_base_url: str = ""
    qwen_api_key: str = ""
    qwen_model: str = ""
    hyde_enabled: bool = True
    hyde_max_output_tokens: int = Field(default=256, ge=32, le=2048)
    retrieval_recall_k: int = Field(default=20, ge=2, le=100)
    retrieval_rrf_k: int = Field(default=60, ge=1, le=1000)
    retrieval_min_evidence: int = Field(default=1, ge=1, le=20)
    retrieval_max_evidence: int = Field(default=8, ge=1, le=20)
    retrieval_score_cliff_ratio: float = Field(default=0.45, ge=0, le=1)
    retrieval_rerank_min_score: float = Field(default=0.40, ge=0, le=1)
    rerank_provider: str = "qianwen"
    rerank_api_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "qwen3-rerank"
    rerank_timeout_seconds: int = Field(default=60, ge=5, le=600)
    embedding_provider: str = "qianwen"
    embedding_api_base_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    )
    embedding_api_key: str = ""
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_timeout_seconds: int = Field(default=60, ge=5, le=600)
    sparse_encoder_version: str = "lexical-hash-v1"
    aliyun_web_mcp_url: str = "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp"
    aliyun_web_mcp_api_key: str = ""
    aliyun_web_mcp_tool_name: str = "web_search"
    web_allowed_domains: str = ""
    agent_max_clarification_rounds: int = Field(default=2, ge=1, le=3)
    agent_answer_max_output_tokens: int = Field(default=1400, ge=256, le=4096)

    embedding_dim: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=10, ge=1, le=256)

    @property
    def resolved_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.rerank_api_key

    @property
    def embedding_version(self) -> str:
        return f"{self.embedding_model}+{self.sparse_encoder_version}"

    @property
    def allowed_domains(self) -> tuple[str, ...]:
        return tuple(domain.strip().lower() for domain in self.web_allowed_domains.split(",") if domain.strip())

    @property
    def external_storage_configured(self) -> bool:
        return bool(self.mongodb_uri and self.milvus_uri and self.minio_endpoint and self.redis_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
