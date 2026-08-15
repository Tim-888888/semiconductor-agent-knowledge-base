from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scripts.export_t9446a_contracts import (
    DEFAULT_OUTPUT,
    export_contracts,
    validate_source_manifests,
)
from semikb.contracts.models import (
    DocumentLifecycle,
    IngestUploadMetadata,
    RestoreDocumentRevisionRequest,
    SourceIndexArtifact,
    SourceIngestionMode,
    SourceIngestionPolicy,
    SourceManifest,
)
from semikb.storage.source_manifests import MongoSourceManifestRepository


class FakeCursor:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int) -> FakeCursor:
        self.documents.sort(key=lambda item: item[key], reverse=direction < 0)
        return self

    def limit(self, limit: int) -> FakeCursor:
        self.documents = self.documents[:limit]
        return self

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.documents)


class FakeManifestCollection:
    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []

    @staticmethod
    def _matches(document: dict[str, Any], selector: dict[str, Any]) -> bool:
        return all(document.get(key) == value for key, value in selector.items())

    def find_one(self, selector: dict[str, Any]) -> dict[str, Any] | None:
        return next(
            (deepcopy(item) for item in self.documents if self._matches(item, selector)),
            None,
        )

    def insert_one(self, document: dict[str, Any]) -> object:
        stored = deepcopy(document)
        stored["_id"] = len(self.documents) + 1
        self.documents.append(stored)
        return object()

    def find(self, selector: dict[str, Any]) -> FakeCursor:
        return FakeCursor(
            [deepcopy(item) for item in self.documents if self._matches(item, selector)]
        )


class FakeDatabase:
    def __init__(self) -> None:
        self.source_manifests = FakeManifestCollection()


class FakeClient:
    def __init__(self) -> None:
        self.database = FakeDatabase()

    def __getitem__(self, name: str) -> FakeDatabase:
        assert name == "semikb"
        return self.database


class FakeFactory:
    def __init__(self) -> None:
        self.client = FakeClient()

    @contextmanager
    def mongodb(self) -> Iterator[FakeClient]:
        yield self.client


def _manifest() -> SourceManifest:
    return SourceManifest.model_validate_json(
        Path("data/source_manifests/semikb-demo-corpus-v1.json").read_text(encoding="utf-8")
    )


def test_frozen_demo_manifest_validates_and_local_hash_matches() -> None:
    result = validate_source_manifests()

    assert result["manifest_count"] == 1
    assert result["source_versions"] == ["semikb.demo.synthetic:1.0.0"]
    assert result["locally_verified_hashes"] == ["data/fixtures/demo_corpus.json"]


def test_tabular_policy_forbids_raw_rows_and_requires_profile_and_tool() -> None:
    with pytest.raises(ValidationError, match="Raw tabular rows"):
        SourceIngestionPolicy(
            mode=SourceIngestionMode.TABULAR_PROFILE_AND_TOOL,
            raw_row_vectorization=True,
            index_artifacts=[
                SourceIndexArtifact.DATA_DICTIONARY,
                SourceIndexArtifact.DATASET_PROFILE,
            ],
            analysis_tool_required=True,
        )

    with pytest.raises(ValidationError, match="tabular_profile_and_tool requires"):
        SourceIngestionPolicy(
            mode=SourceIngestionMode.TABULAR_PROFILE_AND_TOOL,
            index_artifacts=[SourceIndexArtifact.DATA_DICTIONARY],
            analysis_tool_required=False,
        )

    policy = SourceIngestionPolicy(
        mode=SourceIngestionMode.TABULAR_PROFILE_AND_TOOL,
        index_artifacts=[
            SourceIndexArtifact.DATA_DICTIONARY,
            SourceIndexArtifact.DATASET_PROFILE,
            SourceIndexArtifact.ANALYSIS_REPORT,
        ],
        analysis_tool_required=True,
    )
    assert policy.raw_row_vectorization is False


def test_source_link_is_optional_for_legacy_uploads_but_atomic_when_present() -> None:
    legacy = IngestUploadMetadata(
        document_id="LEGACY-001",
        revision="R1",
        title="Legacy document",
        document_type="sop",
    )
    assert legacy.source_id is None

    with pytest.raises(ValidationError, match="must be provided together"):
        IngestUploadMetadata(
            document_id="PUBLIC-001",
            revision="R1",
            title="Public document",
            document_type="paper",
            source_id="public.paper",
        )

    linked = IngestUploadMetadata(
        document_id="PUBLIC-001",
        revision="R1",
        title="Public document",
        document_type="paper",
        source_id="public.paper",
        source_manifest_version="1.0.0",
    )
    assert linked.source_manifest_version == "1.0.0"
    assert DocumentLifecycle("withdrawn") is DocumentLifecycle.WITHDRAWN

    restore = RestoreDocumentRevisionRequest(
        request_id="restore-public-001-r1",
        reason="Restore after governance revalidation.",
        target_index_version="v4",
    )
    assert restore.target_index_version == "v4"


def test_source_manifest_versions_are_immutable_and_registration_is_idempotent() -> None:
    factory = FakeFactory()
    repository = MongoSourceManifestRepository(factory, "semikb")  # type: ignore[arg-type]
    manifest = _manifest()

    first = repository.register(manifest)
    second = repository.register(manifest)

    assert first == second
    assert repository.get(manifest.source_id, manifest.manifest_version) == manifest
    assert repository.list() == [manifest]
    assert len(factory.client.database.source_manifests.documents) == 1

    changed = manifest.model_copy(update={"title": "Changed immutable title"})
    with pytest.raises(ValueError, match="different immutable content"):
        repository.register(changed)


def test_frozen_contract_bundle_is_current_and_contains_governance_models() -> None:
    result = export_contracts(DEFAULT_OUTPUT, check=True)
    bundle = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    assert result["status"] == "current"
    assert bundle["contract_version"] == "semikb-source-governance-v1"
    assert set(bundle["schemas"]) == {
        "SourceManifest",
        "KnowledgeDocumentListResponse",
        "KnowledgeDocumentRevisionSummary",
        "WithdrawDocumentRevisionRequest",
        "RestoreDocumentRevisionRequest",
        "DocumentLifecycleOperationRecord",
    }


def test_typescript_contract_includes_withdrawn_and_raw_row_boundary() -> None:
    typescript = Path("web/src/types.ts").read_text(encoding="utf-8")

    assert 'raw_row_vectorization: false;' in typescript
    assert '| "withdrawn";' in typescript
    assert 'export type SourceManifest = {' in typescript
    assert 'export type KnowledgeDocumentListResponse = {' in typescript
    assert 'export type RestoreDocumentRevisionRequest = {' in typescript
    assert 'target_index_version?: string | null;' in typescript
    assert 'affected: AffectedRecordCounts;' in typescript
    assert 'export type DocumentLifecycleOperationRecord = {' in typescript
