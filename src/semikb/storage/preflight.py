"""Explicit external-service preflight for values supplied through ``.env``.

It intentionally never prints credentials, access keys, or full connection strings.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from minio import Minio
from pymilvus import MilvusClient
from pymongo import MongoClient
from redis import Redis

from semikb.config import Settings, get_settings
from semikb.storage.external import ServiceHealth, service_configuration_health


def run_preflight(settings: Settings) -> list[ServiceHealth]:
    """Ping every configured datastore and report a safe, actionable status."""

    results: list[ServiceHealth] = []
    for item in service_configuration_health(settings):
        if not item.configured:
            results.append(item)
            continue
        try:
            if item.name == "mongodb":
                MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=3000).admin.command("ping")
            elif item.name == "milvus":
                client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token or None)
                client.list_collections()
            elif item.name == "minio":
                client = Minio(
                    settings.minio_endpoint,
                    access_key=settings.minio_access_key,
                    secret_key=settings.minio_secret_key,
                    secure=settings.minio_secure,
                )
                client.list_buckets()
            elif item.name == "redis":
                Redis.from_url(settings.redis_url, socket_connect_timeout=3).ping()
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
