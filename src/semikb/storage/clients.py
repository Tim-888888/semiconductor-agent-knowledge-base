"""Centralized factories and lifetime management for external datastore clients."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from minio import Minio
from pymilvus import MilvusClient
from pymongo import MongoClient
from redis import Redis

from semikb.config import Settings


class StorageConfigurationError(RuntimeError):
    """Raised without endpoint or credential values when datastore settings are incomplete."""


STORAGE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "mongodb": ("MONGODB_URI",),
    "milvus": ("MILVUS_URI",),
    "minio": ("MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"),
    "redis": ("REDIS_URL",),
}


def missing_storage_settings(settings: Settings, service: str) -> tuple[str, ...]:
    values = {
        "MONGODB_URI": settings.mongodb_uri,
        "MILVUS_URI": settings.milvus_uri,
        "MINIO_ENDPOINT": settings.minio_endpoint,
        "MINIO_ACCESS_KEY": settings.minio_access_key,
        "MINIO_SECRET_KEY": settings.minio_secret_key,
        "REDIS_URL": settings.redis_url,
    }
    return tuple(name for name in STORAGE_REQUIREMENTS[service] if not values[name])


@dataclass(slots=True)
class StorageClientFactory:
    """Create clients with consistent timeouts and close networked clients deterministically."""

    settings: Settings

    def _require(self, service: str) -> None:
        missing = missing_storage_settings(self.settings, service)
        if missing:
            raise StorageConfigurationError(
                f"{service} configuration is incomplete; missing: {', '.join(missing)}"
            )

    def create_mongodb(self) -> MongoClient:
        self._require("mongodb")
        return MongoClient(
            self.settings.mongodb_uri,
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=3000,
        )

    def create_milvus(self) -> MilvusClient:
        self._require("milvus")
        return MilvusClient(
            uri=self.settings.milvus_uri,
            token=self.settings.milvus_token or None,
            timeout=3,
        )

    def create_minio(self) -> Minio:
        self._require("minio")
        return Minio(
            self.settings.minio_endpoint,
            access_key=self.settings.minio_access_key,
            secret_key=self.settings.minio_secret_key,
            secure=self.settings.minio_secure,
        )

    def create_redis(self) -> Redis:
        self._require("redis")
        return Redis.from_url(
            self.settings.redis_url,
            socket_connect_timeout=3,
            socket_timeout=3,
            decode_responses=True,
        )

    @contextmanager
    def mongodb(self) -> Iterator[MongoClient]:
        client = self.create_mongodb()
        try:
            yield client
        finally:
            client.close()

    @contextmanager
    def milvus(self) -> Iterator[MilvusClient]:
        client = self.create_milvus()
        try:
            yield client
        finally:
            client.close()

    @contextmanager
    def redis(self) -> Iterator[Redis]:
        client = self.create_redis()
        try:
            yield client
        finally:
            client.close()
