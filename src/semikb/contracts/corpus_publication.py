"""Versioned contracts for reviewed corpus publication batches."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semikb.contracts.models import (
    RedistributionPolicy,
    RetrievalPolicy,
    SourceContentOrigin,
    SourceLicenseStatus,
    SourceManifest,
    SourceManifestType,
    new_id,
    utc_now,
)


class StrictPublicationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusPublicationStatus(StrEnum):
    QUEUED = "queued"
    PREFLIGHT = "preflight"
    PUBLISHING = "publishing"
    RECONCILING = "reconciling"
    COMPLETED = "completed"
    FAILED = "failed"


class CorpusPublicationItemStatus(StrEnum):
    PENDING = "pending"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


class CorpusPublicationArtifactKind(StrEnum):
    DOCUMENT = "document"
    TABULAR_PROFILE = "tabular_profile"
    IMAGE_TEXT = "image_text"


class CorpusPublicationReview(StrictPublicationModel):
    request_id: str = Field(min_length=1, max_length=160)
    standardization_job_id: str = Field(min_length=1, max_length=160)
    expected_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_file_ids: list[str] = Field(min_length=1, max_length=500)
    acknowledged_warning_codes: list[str] = Field(default_factory=list, max_length=100)
    source_type: SourceManifestType
    content_origin: SourceContentOrigin = SourceContentOrigin.REAL
    source_url: str = Field(min_length=1, max_length=2000)
    license_name: str = Field(min_length=1, max_length=200)
    license_status: SourceLicenseStatus
    redistribution_policy: RedistributionPolicy
    license_notes: str = Field(default="", max_length=4000)
    access_scope_key: str = Field(min_length=1, max_length=160)
    review_note: str = Field(min_length=12, max_length=2000)
    retrieval_policy: RetrievalPolicy = RetrievalPolicy.STANDARD

    @model_validator(mode="after")
    def validate_review(self) -> CorpusPublicationReview:
        if len(self.selected_file_ids) != len(set(self.selected_file_ids)):
            raise ValueError("selected_file_ids must be unique.")
        if len(self.acknowledged_warning_codes) != len(
            set(self.acknowledged_warning_codes)
        ):
            raise ValueError("acknowledged_warning_codes must be unique.")
        if self.license_status not in {
            SourceLicenseStatus.VERIFIED,
            SourceLicenseStatus.DECLARED,
        }:
            raise ValueError("Publication requires a reviewed license status.")
        if self.redistribution_policy not in {
            RedistributionPolicy.ALLOWED,
            RedistributionPolicy.RESTRICTED,
        }:
            raise ValueError("Publication requires an explicit redistribution policy.")
        return self


class CorpusPublicationReconciliation(StrictPublicationModel):
    document_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    image_count: int = Field(default=0, ge=0)
    table_count: int = Field(default=0, ge=0)
    vector_count: int = Field(default=0, ge=0)
    object_count: int = Field(default=0, ge=0)
    published_chunk_ids: list[str] = Field(default_factory=list, max_length=5000)
    published_image_ids: list[str] = Field(default_factory=list, max_length=1000)
    published_table_ids: list[str] = Field(default_factory=list, max_length=1000)
    passed: bool = False
    warning_codes: list[str] = Field(default_factory=list)


class CorpusPublicationItem(StrictPublicationModel):
    file_id: str
    relative_path: str
    artifact_kind: CorpusPublicationArtifactKind
    document_id: str
    revision: str
    title: str
    document_type: str
    status: CorpusPublicationItemStatus = CorpusPublicationItemStatus.PENDING
    ingestion_job_id: str | None = None
    error_code: str | None = None
    safe_error_summary: str | None = None
    reconciliation: CorpusPublicationReconciliation | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class CorpusPublicationEvent(StrictPublicationModel):
    event_id: str = Field(default_factory=lambda: new_id("corpus_pub_evt"))
    status: CorpusPublicationStatus
    message: str = Field(min_length=1, max_length=1000)
    progress: int = Field(default=0, ge=0, le=100)
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class CorpusPublicationBatch(StrictPublicationModel):
    batch_schema_version: Literal["semikb-corpus-publication-batch-v1"] = (
        "semikb-corpus-publication-batch-v1"
    )
    batch_id: str = Field(default_factory=lambda: new_id("corpus_pub"))
    review: CorpusPublicationReview
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: CorpusPublicationStatus = CorpusPublicationStatus.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    attempt: int = Field(default=1, ge=1)
    source_manifest: SourceManifest
    items: list[CorpusPublicationItem] = Field(min_length=1)
    published_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    safe_error_summary: str | None = None
    events: list[CorpusPublicationEvent] = Field(default_factory=list)
    created_by: str = Field(min_length=1, max_length=160)
    worker_task_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
