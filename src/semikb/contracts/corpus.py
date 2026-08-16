"""Versioned contracts for generic corpus standardization and review."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from semikb.contracts.models import ObjectRef, new_id, utc_now


class StrictCorpusModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusKind(StrEnum):
    AUTO = "auto"
    DOCUMENT_COLLECTION = "document_collection"
    TABULAR_DATASET = "tabular_dataset"
    IMAGE_CORPUS = "image_corpus"
    MIXED = "mixed"


class CorpusFileRole(StrEnum):
    DOCUMENT = "document"
    TABLE = "table"
    IMAGE = "image"
    LABEL = "label"
    ARCHIVE = "archive"
    UNSUPPORTED = "unsupported"


class CorpusRelationType(StrEnum):
    LABELS = "labels"
    DESCRIBES = "describes"
    COMPANION = "companion"
    DERIVED_FROM = "derived_from"


class CorpusStandardizationStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    SNAPSHOTTING = "snapshotting"
    INVENTORYING = "inventorying"
    STANDARDIZING = "standardizing"
    REVIEW_REQUIRED = "review_required"
    FAILED = "failed"


class CorpusRoleRule(StrictCorpusModel):
    glob: str = Field(min_length=1, max_length=500)
    role: CorpusFileRole


class CorpusRelationRule(StrictCorpusModel):
    from_glob: str = Field(min_length=1, max_length=500)
    to_glob: str = Field(min_length=1, max_length=500)
    relation_type: CorpusRelationType


class CorpusFileAnnotation(StrictCorpusModel):
    path: str = Field(min_length=1, max_length=1000)
    role: CorpusFileRole | None = None
    description: str = Field(default="", max_length=2000)
    tabular_delimiter: str | None = Field(default=None, min_length=1, max_length=16)
    tabular_has_header: bool = True

    @field_validator("tabular_delimiter")
    @classmethod
    def validate_delimiter(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value in {"whitespace", "tab"} or len(value) == 1:
            return value
        raise ValueError("tabular_delimiter must be one character, 'tab', or 'whitespace'.")

    @model_validator(mode="after")
    def validate_tabular_options(self) -> CorpusFileAnnotation:
        if self.tabular_delimiter is not None and self.role not in {
            CorpusFileRole.TABLE,
            CorpusFileRole.LABEL,
        }:
            raise ValueError("Tabular options require role table or label.")
        return self


class CorpusProfile(StrictCorpusModel):
    profile_schema_version: Literal["semikb-corpus-profile-v1"] = (
        "semikb-corpus-profile-v1"
    )
    corpus_kind: CorpusKind = CorpusKind.AUTO
    include_globs: list[str] = Field(default_factory=lambda: ["**/*", "*"])
    exclude_globs: list[str] = Field(
        default_factory=lambda: ["**/.DS_Store", "**/Thumbs.db", "**/__MACOSX/**"]
    )
    role_rules: list[CorpusRoleRule] = Field(default_factory=list)
    relation_rules: list[CorpusRelationRule] = Field(default_factory=list)
    tabular_sample_rows: int = Field(default=200, ge=10, le=2000)
    tabular_max_columns: int = Field(default=256, ge=1, le=1024)
    generate_image_text: bool = True

    @field_validator("include_globs", "exclude_globs")
    @classmethod
    def validate_globs(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 500 for item in value):
            raise ValueError("Glob patterns must be non-empty and at most 500 characters.")
        return value


class CorpusSidecar(StrictCorpusModel):
    sidecar_schema_version: Literal["semikb-corpus-sidecar-v1"] = (
        "semikb-corpus-sidecar-v1"
    )
    profile: CorpusProfile = Field(default_factory=CorpusProfile)
    files: list[CorpusFileAnnotation] = Field(default_factory=list)
    relations: list[CorpusRelationRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_annotations(self) -> CorpusSidecar:
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Sidecar file annotations must use unique paths.")
        return self


class CorpusStandardizationMetadata(StrictCorpusModel):
    metadata_schema_version: Literal["semikb-corpus-metadata-v1"] = (
        "semikb-corpus-metadata-v1"
    )
    corpus_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._-]+$")
    snapshot_version: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=500)
    source_kind: str = Field(default="user_upload", min_length=1, max_length=160)
    source_uri: str = Field(default="", max_length=2000)
    source_license: str = Field(default="unknown", min_length=1, max_length=200)
    use_restrictions: str = Field(default="", max_length=4000)
    access_scope_key: str = Field(default="", max_length=160)
    corpus_kind: CorpusKind = CorpusKind.AUTO


class CorpusStandardizationEvent(StrictCorpusModel):
    event_id: str = Field(default_factory=lambda: new_id("corpus_evt"))
    status: CorpusStandardizationStatus
    message: str = Field(min_length=1, max_length=1000)
    progress: int = Field(default=0, ge=0, le=100)
    attempt: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)


class TabularColumnProfile(StrictCorpusModel):
    name: str = Field(max_length=500)
    inferred_types: dict[str, int] = Field(default_factory=dict)
    non_empty_count: int = Field(default=0, ge=0)
    empty_count: int = Field(default=0, ge=0)
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_mean: float | None = None


class TabularSheetProfile(StrictCorpusModel):
    name: str = Field(max_length=500)
    observed_rows: int = Field(default=0, ge=0)
    column_count: int = Field(default=0, ge=0)
    sample_truncated: bool = False
    columns_truncated: bool = False
    columns: list[TabularColumnProfile] = Field(default_factory=list)


class TabularDataProfile(StrictCorpusModel):
    profile_schema_version: Literal["semikb-tabular-profile-v1"] = (
        "semikb-tabular-profile-v1"
    )
    sheets: list[TabularSheetProfile] = Field(default_factory=list)
    raw_rows_vectorized: Literal[False] = False


class CorpusFileManifest(StrictCorpusModel):
    file_id: str
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    content_type: str
    role: CorpusFileRole
    source_format: str | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    raw_ref: ObjectRef
    standardized_ref: ObjectRef | None = None
    tabular_profile: TabularDataProfile | None = None
    warning_codes: list[str] = Field(default_factory=list)
    description: str = ""
    tabular_delimiter: str | None = None
    tabular_has_header: bool = True


class CorpusFileRelation(StrictCorpusModel):
    relation_id: str = Field(default_factory=lambda: new_id("corpus_rel"))
    from_file_id: str
    to_file_id: str
    relation_type: CorpusRelationType
    source: Literal["sidecar"] = "sidecar"


class CorpusStandardizationReport(StrictCorpusModel):
    report_schema_version: Literal["semikb-corpus-report-v1"] = (
        "semikb-corpus-report-v1"
    )
    corpus_id: str
    snapshot_version: str
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inferred_corpus_kind: CorpusKind
    files: list[CorpusFileManifest] = Field(default_factory=list)
    relations: list[CorpusFileRelation] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)


class CorpusStandardizationJob(StrictCorpusModel):
    job_id: str = Field(default_factory=lambda: new_id("corpus"))
    metadata: CorpusStandardizationMetadata
    sidecar: CorpusSidecar = Field(default_factory=CorpusSidecar)
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    idempotency_key: str
    source_refs: list[ObjectRef] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    status: CorpusStandardizationStatus = CorpusStandardizationStatus.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    attempt: int = Field(default=1, ge=1)
    files_count: int = Field(default=0, ge=0)
    documents_count: int = Field(default=0, ge=0)
    tables_count: int = Field(default=0, ge=0)
    images_count: int = Field(default=0, ge=0)
    unsupported_count: int = Field(default=0, ge=0)
    report: CorpusStandardizationReport | None = None
    report_ref: ObjectRef | None = None
    error_code: str | None = None
    safe_error_summary: str | None = None
    created_by: str = Field(min_length=1, max_length=160)
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    events: list[CorpusStandardizationEvent] = Field(default_factory=list)


class CorpusUploadedFile(StrictCorpusModel):
    relative_path: str = Field(min_length=1, max_length=1000)
    content_type: str = Field(default="application/octet-stream", max_length=200)
    content: bytes = Field(min_length=1)


class CorpusStandardizationError(RuntimeError):
    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
