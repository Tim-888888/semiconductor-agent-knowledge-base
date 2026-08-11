"""Optional external-service checks. Demo mode never calls these adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from semikb.config import Settings


@dataclass(slots=True)
class ServiceHealth:
    name: str
    configured: bool
    reachable: bool | None
    detail: str


def service_configuration_health(settings: Settings) -> list[ServiceHealth]:
    """Report configuration readiness without logging endpoints or credentials."""

    return [
        ServiceHealth("mongodb", bool(settings.mongodb_uri), None, "URI configured"),
        ServiceHealth("milvus", bool(settings.milvus_uri), None, "URI configured"),
        ServiceHealth("minio", bool(settings.minio_endpoint), None, "endpoint configured"),
        ServiceHealth("redis", bool(settings.redis_url), None, "URL configured"),
        ServiceHealth("mineru", bool(settings.mineru_api_base_url and settings.mineru_api_key), None, "API configured"),
        ServiceHealth("llm", bool(settings.llm_api_base_url and settings.llm_api_key), None, "API configured"),
        ServiceHealth(
            "aliyun_web_mcp",
            bool(settings.aliyun_web_mcp_url and settings.aliyun_web_mcp_api_key),
            None,
            "MCP configured",
        ),
    ]


def health_payload(settings: Settings) -> list[dict[str, object]]:
    return [asdict(item) for item in service_configuration_health(settings)]
