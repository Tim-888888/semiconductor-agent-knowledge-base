"""Immutable MongoDB registry for versioned source manifests."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pymongo.errors import DuplicateKeyError

from semikb.contracts.models import SourceManifest, SourceManifestStatus, SourceManifestType
from semikb.storage.clients import StorageClientFactory


def source_manifest_checksum(manifest: SourceManifest) -> str:
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _without_mongo_id(document: dict[str, Any] | None) -> dict[str, Any] | None:
    if document is None:
        return None
    return {
        key: value
        for key, value in document.items()
        if key not in {"_id", "manifest_checksum"}
    }


class MongoSourceManifestRepository:
    """Registers immutable manifest versions; updates require a new version."""

    def __init__(self, factory: StorageClientFactory, database_name: str) -> None:
        self._factory = factory
        self._database_name = database_name

    def register(self, manifest: SourceManifest) -> SourceManifest:
        selector = {
            "source_id": manifest.source_id,
            "manifest_version": manifest.manifest_version,
        }
        checksum = source_manifest_checksum(manifest)
        document = {
            **manifest.model_dump(mode="python"),
            "manifest_checksum": checksum,
        }
        with self._factory.mongodb() as client:
            collection = client[self._database_name].source_manifests
            existing = collection.find_one(selector)
            if existing is None:
                try:
                    collection.insert_one(document)
                    return manifest
                except DuplicateKeyError:
                    existing = collection.find_one(selector)
            if existing is None:
                raise RuntimeError("Source manifest registration lost a concurrent insert.")
            existing_manifest = SourceManifest.model_validate(_without_mongo_id(existing))
            existing_checksum = existing.get("manifest_checksum") or source_manifest_checksum(
                existing_manifest
            )
            if existing_checksum != checksum:
                raise ValueError(
                    "The source manifest version already exists with different immutable content."
                )
            return existing_manifest

    def get(self, source_id: str, manifest_version: str) -> SourceManifest | None:
        with self._factory.mongodb() as client:
            stored = client[self._database_name].source_manifests.find_one(
                {"source_id": source_id, "manifest_version": manifest_version}
            )
        clean = _without_mongo_id(stored)
        return SourceManifest.model_validate(clean) if clean else None

    def list(
        self,
        *,
        status: SourceManifestStatus | None = None,
        source_type: SourceManifestType | None = None,
        limit: int = 200,
    ) -> list[SourceManifest]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        selector: dict[str, str] = {}
        if status is not None:
            selector["status"] = status.value
        if source_type is not None:
            selector["source_type"] = source_type.value
        with self._factory.mongodb() as client:
            records = list(
                client[self._database_name]
                .source_manifests.find(selector)
                .sort("created_at", -1)
                .limit(limit)
            )
        return [
            SourceManifest.model_validate(_without_mongo_id(record)) for record in records
        ]
