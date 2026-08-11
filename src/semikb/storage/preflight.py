"""Explicit external-service preflight for values supplied through ``.env``.

It intentionally never prints credentials, access keys, or full connection strings.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from semikb.config import Settings, get_settings
from semikb.storage.clients import StorageClientFactory
from semikb.storage.external import ServiceHealth, service_configuration_health


def run_preflight(settings: Settings) -> list[ServiceHealth]:
    """Ping every configured datastore and report a safe, actionable status."""

    results: list[ServiceHealth] = []
    factory = StorageClientFactory(settings)
    for item in service_configuration_health(settings):
        if not item.configured:
            results.append(item)
            continue
        try:
            if item.name == "mongodb":
                with factory.mongodb() as client:
                    client.admin.command("ping")
            elif item.name == "milvus":
                with factory.milvus() as client:
                    client.list_collections()
            elif item.name == "minio":
                client = factory.create_minio()
                client.list_buckets()
            elif item.name == "redis":
                with factory.redis() as client:
                    client.ping()
            else:
                results.append(item)
                continue
            results.append(ServiceHealth(item.name, True, True, "connection verified"))
        except Exception as exc:  # pragma: no cover - depends on user infrastructure
            results.append(ServiceHealth(item.name, True, False, f"{type(exc).__name__}: check endpoint and credentials"))
    return results


def main() -> None:
    settings = get_settings()
    print(json.dumps([asdict(item) for item in run_preflight(settings)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
