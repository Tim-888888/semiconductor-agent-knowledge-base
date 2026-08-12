"""Generate a server-only .env without copying workstation or SSH credentials."""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path
from urllib.parse import quote

PROVIDER_KEYS = (
    "MINERU_API_BASE_URL",
    "MINERU_API_KEY",
    "MINERU_MODEL_VERSION",
    "MINERU_TIMEOUT_SECONDS",
    "MINERU_POLL_SECONDS",
    "LLM_PRIMARY_PROVIDER",
    "LLM_FALLBACK_PROVIDER",
    "LLM_TIMEOUT_SECONDS",
    "CLOSEAI_BASE_URL",
    "CLOSEAI_API_KEY",
    "CLOSEAI_MODEL",
    "CLOSEAI_REASONING_EFFORT",
    "CLOSEAI_VERBOSITY",
    "QWEN_API_BASE_URL",
    "QWEN_API_KEY",
    "QWEN_MODEL",
    "HYDE_ENABLED",
    "HYDE_MAX_OUTPUT_TOKENS",
    "RETRIEVAL_RECALL_K",
    "RETRIEVAL_RRF_K",
    "RETRIEVAL_MIN_EVIDENCE",
    "RETRIEVAL_MAX_EVIDENCE",
    "RETRIEVAL_SCORE_CLIFF_RATIO",
    "RETRIEVAL_RERANK_MIN_SCORE",
    "RERANK_PROVIDER",
    "RERANK_API_BASE_URL",
    "RERANK_API_KEY",
    "RERANK_MODEL",
    "RERANK_TIMEOUT_SECONDS",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_API_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_OUTPUT_TYPE",
    "EMBEDDING_TIMEOUT_SECONDS",
    "SPARSE_ENCODER_VERSION",
    "EMBEDDING_DIM",
    "EMBEDDING_BATCH_SIZE",
    "ALIYUN_WEB_MCP_URL",
    "ALIYUN_WEB_MCP_API_KEY",
    "ALIYUN_WEB_MCP_TOOL_NAME",
    "WEB_ALLOWED_DOMAINS",
    "AGENT_MAX_CLARIFICATION_ROUNDS",
    "AGENT_ANSWER_MAX_OUTPUT_TOKENS",
)

DEFAULTS = {
    "MINERU_MODEL_VERSION": "vlm",
    "MINERU_TIMEOUT_SECONDS": "900",
    "MINERU_POLL_SECONDS": "3",
    "LLM_PRIMARY_PROVIDER": "closeai",
    "LLM_FALLBACK_PROVIDER": "qwen",
    "LLM_TIMEOUT_SECONDS": "60",
    "CLOSEAI_MODEL": "gpt-5.6-luna",
    "CLOSEAI_REASONING_EFFORT": "none",
    "CLOSEAI_VERBOSITY": "low",
    "QWEN_MODEL": "qwen-flash",
    "HYDE_ENABLED": "true",
    "HYDE_MAX_OUTPUT_TOKENS": "256",
    "RETRIEVAL_RECALL_K": "20",
    "RETRIEVAL_RRF_K": "60",
    "RETRIEVAL_MIN_EVIDENCE": "1",
    "RETRIEVAL_MAX_EVIDENCE": "8",
    "RETRIEVAL_SCORE_CLIFF_RATIO": "0.45",
    "RETRIEVAL_RERANK_MIN_SCORE": "0.40",
    "RERANK_PROVIDER": "qianwen",
    "RERANK_MODEL": "qwen3-rerank",
    "RERANK_TIMEOUT_SECONDS": "60",
    "EMBEDDING_PROVIDER": "qianwen",
    "EMBEDDING_MODEL": "qwen3.7-text-embedding",
    "EMBEDDING_OUTPUT_TYPE": "dense&sparse",
    "EMBEDDING_TIMEOUT_SECONDS": "60",
    "SPARSE_ENCODER_VERSION": "qwen3.7-text-embedding-sparse-v1",
    "EMBEDDING_DIM": "1024",
    "EMBEDDING_BATCH_SIZE": "10",
    "ALIYUN_WEB_MCP_TOOL_NAME": "web_search",
    "AGENT_MAX_CLARIFICATION_ROUNDS": "2",
    "AGENT_ANSWER_MAX_OUTPUT_TOKENS": "1400",
}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _secret(bytes_count: int = 24) -> str:
    return secrets.token_hex(bytes_count)


def build_server_values(source: dict[str, str]) -> dict[str, str]:
    mongo_user = "semikb_app"
    mongo_password = _secret()
    minio_user = f"semikb{secrets.token_hex(6)}"
    minio_password = _secret()
    redis_password = _secret()
    milvus_password = _secret()
    values = {
        "APP_ENV": "production",
        "DEMO_MODE": "false",
        "JWT_SECRET": _secret(32),
        "DEMO_ACCESS_KEY": _secret(16),
        "MAX_UPLOAD_MIB": "100",
        "SEMIKB_DATA_ROOT": "/opt/semiconductor-agent-knowledge-base-data",
        "SEMIKB_BACKUP_ROOT": "/opt/semiconductor-agent-knowledge-base-backups",
        "MONGO_ROOT_USERNAME": "semikb_root",
        "MONGO_ROOT_PASSWORD": _secret(),
        "MONGO_APP_USERNAME": mongo_user,
        "MONGO_APP_PASSWORD": mongo_password,
        "MONGODB_URI": (
            f"mongodb://{quote(mongo_user, safe='')}:{quote(mongo_password, safe='')}"
            "@mongodb:27017/semikb?authSource=semikb"
        ),
        "MONGODB_DATABASE": "semikb",
        "MILVUS_ROOT_PASSWORD": milvus_password,
        "MILVUS_MINIO_ROOT_USER": f"milvus{secrets.token_hex(6)}",
        "MILVUS_MINIO_ROOT_PASSWORD": _secret(),
        "MILVUS_URI": "http://milvus:19530",
        "MILVUS_TOKEN": f"root:{milvus_password}",
        "MILVUS_INDEX_VERSION": "v4",
        "MILVUS_SEARCH_COLLECTION": "semikb_chunks_active",
        "MILVUS_REQUIRE_ACTIVE_ALIAS": "true",
        "MINIO_ROOT_USER": minio_user,
        "MINIO_ROOT_PASSWORD": minio_password,
        "MINIO_ENDPOINT": "minio:9000",
        "MINIO_ACCESS_KEY": minio_user,
        "MINIO_SECRET_KEY": minio_password,
        "MINIO_SECURE": "false",
        "MINIO_PUBLIC_BASE_URL": "/objects",
        "REDIS_PASSWORD": redis_password,
        "REDIS_URL": f"redis://:{redis_password}@redis:6379/0",
    }
    for key in PROVIDER_KEYS:
        value = source.get(key, DEFAULTS.get(key, ""))
        if value:
            values[key] = value
    return values


def render_env(values: dict[str, str]) -> str:
    lines = [
        "# Generated for the full-stack single-node deployment. Do not commit this file.",
        "# Regenerate for a new server instead of reusing infrastructure passwords.",
    ]
    lines.extend(f"{key}={value}" for key, value in values.items())
    return "\n".join(lines) + "\n"


def generate(source_path: Path, output_path: Path, *, force: bool = False) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    source = read_env(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_env(build_server_values(source)), encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/runtime/deployment/.env.server"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    generate(args.source, args.output, force=args.force)
    print(f"Server environment generated at {args.output}; secret values were not printed.")


if __name__ == "__main__":
    main()
