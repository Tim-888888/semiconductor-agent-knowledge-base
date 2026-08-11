"""Optional external-service checks. Demo mode never calls these adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from semikb.config import Settings
from semikb.storage.clients import STORAGE_REQUIREMENTS, missing_storage_settings


@dataclass(slots=True)
class ServiceHealth:
    name: str
    configured: bool
    reachable: bool | None
    detail: str


def service_configuration_health(settings: Settings) -> list[ServiceHealth]:
    """Report configuration readiness without logging endpoints or credentials."""

    health: list[ServiceHealth] = []
    for service in STORAGE_REQUIREMENTS:
        missing = missing_storage_settings(settings, service)
        health.append(
            ServiceHealth(
                service,
                not missing,
                None,
                "configuration present" if not missing else f"not configured: {', '.join(missing)}",
            )
        )

    api_requirements = {
        "mineru": {
            "MINERU_API_BASE_URL": settings.mineru_api_base_url,
            "MINERU_API_KEY": settings.mineru_api_key,
        },
        "llm": {
            "LLM_API_BASE_URL": settings.llm_api_base_url,
            "LLM_API_KEY": settings.llm_api_key,
            "LLM_MODEL": settings.llm_model,
        },
        "aliyun_web_mcp": {
            "ALIYUN_WEB_MCP_URL": settings.aliyun_web_mcp_url,
            "ALIYUN_WEB_MCP_API_KEY": settings.aliyun_web_mcp_api_key,
            "ALIYUN_WEB_MCP_TOOL_NAME": settings.aliyun_web_mcp_tool_name,
        },
    }
    for service, requirements in api_requirements.items():
        missing = tuple(name for name, value in requirements.items() if not value)
        health.append(
            ServiceHealth(
                service,
                not missing,
                None,
                "configuration present" if not missing else f"not configured: {', '.join(missing)}",
            )
        )
    return health


def health_payload(settings: Settings) -> list[dict[str, object]]:
    return [asdict(item) for item in service_configuration_health(settings)]
