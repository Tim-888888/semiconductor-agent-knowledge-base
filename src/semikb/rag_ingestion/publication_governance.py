"""Shared publication gate for first publication and lifecycle restoration."""

from __future__ import annotations

from collections.abc import Sequence

from semikb.contracts.models import (
    ApprovalStatus,
    Chunk,
    ChunkType,
    DocumentRevision,
    ImageAsset,
    RedistributionPolicy,
    SourceIndexArtifact,
    SourceLicenseStatus,
    SourceManifest,
    SourceManifestStatus,
    TableAsset,
)


class PublicationGovernanceError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def validate_publication_governance(
    document: DocumentRevision,
    chunks: Sequence[Chunk],
    images: Sequence[ImageAsset],
    tables: Sequence[TableAsset],
    manifest: SourceManifest | None,
) -> None:
    """Validate reviewed metadata and the representations entering retrieval."""

    if document.approval_status is not ApprovalStatus.APPROVED:
        raise PublicationGovernanceError("revision_not_approved")
    if not document.access_scope_key:
        raise PublicationGovernanceError("access_scope_missing")
    if not document.source_id or not document.source_manifest_version:
        raise PublicationGovernanceError("source_manifest_link_missing")
    if manifest is None:
        raise PublicationGovernanceError("source_manifest_missing")
    if manifest.status is not SourceManifestStatus.APPROVED:
        raise PublicationGovernanceError("source_manifest_not_approved")
    if manifest.source_id != document.source_id or (
        manifest.manifest_version != document.source_manifest_version
    ):
        raise PublicationGovernanceError("source_manifest_link_mismatch")
    if manifest.access_scope_key != document.access_scope_key:
        raise PublicationGovernanceError("source_manifest_scope_mismatch")
    if document.dataset_version != manifest.dataset_version:
        raise PublicationGovernanceError("source_manifest_dataset_mismatch")
    if document.source_license != manifest.license_name:
        raise PublicationGovernanceError("source_manifest_license_mismatch")
    if document.source_license_status is not manifest.license_status:
        raise PublicationGovernanceError("source_manifest_license_status_mismatch")
    if document.redistribution_policy is not manifest.redistribution_policy:
        raise PublicationGovernanceError("source_manifest_redistribution_mismatch")
    if manifest.license_status not in {
        SourceLicenseStatus.VERIFIED,
        SourceLicenseStatus.DECLARED,
    }:
        raise PublicationGovernanceError("source_license_requires_review")
    if manifest.redistribution_policy not in {
        RedistributionPolicy.ALLOWED,
        RedistributionPolicy.RESTRICTED,
    }:
        raise PublicationGovernanceError("source_redistribution_requires_review")
    if document.source_hash != document.source_ref.sha256:
        raise PublicationGovernanceError("source_object_hash_mismatch")
    if manifest.source_snapshot_ref is not None and (
        manifest.source_snapshot_ref.sha256 != manifest.source_hash
    ):
        raise PublicationGovernanceError("source_manifest_snapshot_hash_mismatch")

    actual_artifacts = set(document.source_index_artifacts)
    if not actual_artifacts:
        if any(chunk.chunk_type is not ChunkType.IMAGE_TEXT for chunk in chunks):
            actual_artifacts.add(SourceIndexArtifact.DOCUMENT_CHUNKS)
        if images or any(chunk.chunk_type is ChunkType.IMAGE_TEXT for chunk in chunks):
            actual_artifacts.add(SourceIndexArtifact.IMAGE_TEXT)
    declared_artifacts = set(manifest.ingestion_policy.index_artifacts)
    if not actual_artifacts.issubset(declared_artifacts):
        raise PublicationGovernanceError("source_ingestion_policy_mismatch")
    if chunks and not actual_artifacts:
        raise PublicationGovernanceError("source_ingestion_policy_empty")
    if tables and manifest.ingestion_policy.raw_row_vectorization:
        raise PublicationGovernanceError("raw_table_rows_cannot_be_vectorized")
