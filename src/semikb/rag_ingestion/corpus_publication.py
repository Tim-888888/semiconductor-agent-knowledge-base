"""Reviewed publication of generic corpus standardization outputs."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Protocol

from semikb.contracts.corpus import (
    CorpusFileManifest,
    CorpusFileRole,
    CorpusStandardizationJob,
    CorpusStandardizationStatus,
    TabularDataProfile,
)
from semikb.contracts.corpus_publication import (
    CorpusPublicationArtifactKind,
    CorpusPublicationBatch,
    CorpusPublicationEvent,
    CorpusPublicationItem,
    CorpusPublicationItemStatus,
    CorpusPublicationReview,
    CorpusPublicationStatus,
)
from semikb.contracts.models import (
    ApprovalStatus,
    DocumentLifecycle,
    IngestionStatus,
    SourceExpectedAssets,
    SourceIndexArtifact,
    SourceIngestionMode,
    SourceIngestionPolicy,
    SourceManifest,
    SourceManifestStatus,
)
from semikb.rag_ingestion.service import IngestionService


class CorpusPublicationStore(Protocol):
    def create_or_get(self, batch: CorpusPublicationBatch) -> CorpusPublicationBatch: ...

    def save(self, batch: CorpusPublicationBatch) -> CorpusPublicationBatch: ...

    def get(self, batch_id: str) -> CorpusPublicationBatch | None: ...

    def list(self) -> list[CorpusPublicationBatch]: ...

    def claim(self, batch_id: str, execution_id: str | None) -> CorpusPublicationBatch | None: ...

    def prepare_retry(self, batch_id: str) -> CorpusPublicationBatch: ...


class CorpusStandardizationReadStore(Protocol):
    def get(self, job_id: str) -> CorpusStandardizationJob | None: ...

    def load_object(self, object_ref): ...


class CorpusPublicationError(RuntimeError):
    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class CorpusPublicationService:
    """Publishes only explicitly reviewed, standardized corpus artifacts."""

    def __init__(
        self,
        repository: CorpusPublicationStore,
        corpus_store: CorpusStandardizationReadStore,
        ingestion: IngestionService,
    ) -> None:
        self.repository = repository
        self.corpus_store = corpus_store
        self.ingestion = ingestion

    def submit(
        self,
        review: CorpusPublicationReview,
        *,
        created_by: str,
    ) -> CorpusPublicationBatch:
        job = self._reviewed_job(review)
        assert job.report is not None
        by_id = {item.file_id: item for item in job.report.files}
        selected = [by_id[file_id] for file_id in review.selected_file_ids]
        manifest = self._build_source_manifest(job, review, selected, created_by)
        items = [self._build_item(job, item) for item in selected]
        fingerprint = self._fingerprint(review, manifest, items)
        batch = CorpusPublicationBatch(
            review=review,
            request_fingerprint=fingerprint,
            source_manifest=manifest,
            items=items,
            created_by=created_by,
            events=[
                CorpusPublicationEvent(
                    status=CorpusPublicationStatus.QUEUED,
                    message="Reviewed corpus publication batch accepted.",
                )
            ],
        )
        return self.repository.create_or_get(batch)

    def get(self, batch_id: str) -> CorpusPublicationBatch | None:
        return self.repository.get(batch_id)

    def list(self) -> list[CorpusPublicationBatch]:
        return self.repository.list()

    def prepare_retry(self, batch_id: str) -> CorpusPublicationBatch:
        current = self._require_batch(batch_id)
        reconciliation_failures = {
            item.file_id
            for item in current.items
            if item.status is CorpusPublicationItemStatus.FAILED
            and item.error_code == "CROSS_STORE_RECONCILIATION_FAILED"
        }
        batch = self.repository.prepare_retry(batch_id)
        for item in batch.items:
            if item.file_id not in reconciliation_failures:
                continue
            suffix = f"-retry{batch.attempt}"
            item.revision = f"{item.revision[: max(1, 64 - len(suffix))]}{suffix}"
            item.ingestion_job_id = None
            item.reconciliation = None
        return self.repository.save(batch) if reconciliation_failures else batch

    def mark_queue_submission_failed(self, batch_id: str) -> CorpusPublicationBatch:
        batch = self._require_batch(batch_id)
        return self._finish_failed(
            batch,
            "QUEUE_SUBMISSION_FAILED",
            "Task queue unavailable; unpublished items remain outside active retrieval.",
        )

    def process(
        self,
        batch_id: str,
        *,
        execution_id: str | None = None,
    ) -> CorpusPublicationBatch:
        claimed = self.repository.claim(batch_id, execution_id)
        if claimed is None:
            return self._require_batch(batch_id)
        batch = claimed
        try:
            job = self._reviewed_job(batch.review)
            self._event(
                batch,
                CorpusPublicationStatus.PREFLIGHT,
                "Snapshot, selected artifacts, warnings and publication policy revalidated.",
                10,
            )
            self.ingestion.store.register_source_manifest(batch.source_manifest)
            batch.status = CorpusPublicationStatus.PUBLISHING
            self._event(
                batch,
                CorpusPublicationStatus.PUBLISHING,
                "Publishing reviewed standardized artifacts through the governed ingestion path.",
                20,
            )
            selected = {item.file_id: item for item in job.report.files} if job.report else {}
            pending = [
                item
                for item in batch.items
                if item.status is not CorpusPublicationItemStatus.PUBLISHED
            ]
            for index, item in enumerate(pending, start=1):
                source = selected[item.file_id]
                self._publish_item(batch, item, job, source)
                batch.progress = 20 + int(60 * index / max(1, len(pending)))
                batch.published_count = sum(
                    entry.status is CorpusPublicationItemStatus.PUBLISHED
                    for entry in batch.items
                )
                batch.failed_count = sum(
                    entry.status is CorpusPublicationItemStatus.FAILED
                    for entry in batch.items
                )
                self.repository.save(batch)

            batch.status = CorpusPublicationStatus.RECONCILING
            self._event(
                batch,
                CorpusPublicationStatus.RECONCILING,
                "Cross-store publication read-back completed for every successful item.",
                90,
            )
            if batch.failed_count:
                return self._finish_failed(
                    batch,
                    "PUBLICATION_ITEMS_FAILED",
                    "One or more reviewed artifacts remain unpublished; retry only failed items.",
                )
            batch.status = CorpusPublicationStatus.COMPLETED
            batch.progress = 100
            batch.finished_at = datetime.now(UTC)
            batch.error_code = None
            batch.safe_error_summary = None
            self._event(
                batch,
                CorpusPublicationStatus.COMPLETED,
                "All reviewed artifacts were published and reconciled.",
                100,
            )
            return self.repository.save(batch)
        except CorpusPublicationError as exc:
            return self._finish_failed(batch, exc.code, exc.safe_message)
        except Exception as exc:
            return self._finish_failed(
                batch,
                type(exc).__name__.upper(),
                "Corpus publication failed safely; unfinished items remain outside active retrieval.",
            )

    def _publish_item(
        self,
        batch: CorpusPublicationBatch,
        item: CorpusPublicationItem,
        job: CorpusStandardizationJob,
        source: CorpusFileManifest,
    ) -> None:
        item.status = CorpusPublicationItemStatus.PUBLISHING
        item.updated_at = datetime.now(UTC)
        self.repository.save(batch)
        try:
            content, content_type, images = self._publication_payload(source, job, item)
            metadata = self._ingestion_metadata(batch, item, job, source, images)
            ingest_job = self.ingestion.submit_file(
                f"{item.document_id}-{item.revision}.md",
                content,
                metadata,
                batch.created_by,
                content_type=content_type,
            )
            if ingest_job.status is IngestionStatus.FAILED:
                ingest_job = self.ingestion.retry(ingest_job.job_id)
            elif ingest_job.status is IngestionStatus.QUEUED:
                ingest_job = self.ingestion.process(ingest_job.job_id)
            item.ingestion_job_id = ingest_job.job_id
            if ingest_job.status is not IngestionStatus.PUBLISHED:
                raise CorpusPublicationError(
                    ingest_job.error_code or "INGESTION_NOT_PUBLISHED",
                    ingest_job.safe_error_summary
                    or "The reviewed artifact did not pass governed ingestion.",
                )
            reconciliation = self.ingestion.store.reconcile_published_document(
                item.document_id,
                item.revision,
            )
            item.reconciliation = reconciliation
            if not reconciliation.passed:
                self.ingestion.store.compensate_document(
                    item.document_id,
                    item.revision,
                )
                raise CorpusPublicationError(
                    "CROSS_STORE_RECONCILIATION_FAILED",
                    "Published projections failed read-back and were isolated from active retrieval.",
                )
            item.status = CorpusPublicationItemStatus.PUBLISHED
            item.error_code = None
            item.safe_error_summary = None
        except CorpusPublicationError as exc:
            item.status = CorpusPublicationItemStatus.FAILED
            item.error_code = exc.code
            item.safe_error_summary = exc.safe_message
        except Exception as exc:
            item.status = CorpusPublicationItemStatus.FAILED
            item.error_code = type(exc).__name__.upper()
            item.safe_error_summary = "The reviewed artifact failed before publication completed."
        finally:
            item.updated_at = datetime.now(UTC)

    def _reviewed_job(self, review: CorpusPublicationReview) -> CorpusStandardizationJob:
        job = self.corpus_store.get(review.standardization_job_id)
        if job is None:
            raise CorpusPublicationError(
                "STANDARDIZATION_JOB_NOT_FOUND",
                "The reviewed standardization job does not exist.",
            )
        if job.status is not CorpusStandardizationStatus.REVIEW_REQUIRED or job.report is None:
            raise CorpusPublicationError(
                "STANDARDIZATION_NOT_REVIEWABLE",
                "Only completed standardization reports can be published.",
            )
        if job.snapshot_hash != review.expected_snapshot_hash:
            raise CorpusPublicationError(
                "SNAPSHOT_HASH_MISMATCH",
                "The corpus snapshot changed after review.",
            )
        by_id = {item.file_id: item for item in job.report.files}
        missing = sorted(set(review.selected_file_ids).difference(by_id))
        if missing:
            raise CorpusPublicationError(
                "SELECTED_FILE_NOT_FOUND",
                "One or more reviewed files are not present in this snapshot.",
            )
        selected = [by_id[file_id] for file_id in review.selected_file_ids]
        if any(
            item.role in {CorpusFileRole.ARCHIVE, CorpusFileRole.UNSUPPORTED}
            or item.standardized_ref is None
            for item in selected
        ):
            raise CorpusPublicationError(
                "SELECTED_FILE_NOT_PUBLISHABLE",
                "Only standardized document, table, label, or image artifacts can be published.",
            )
        required_warnings = set(job.report.warning_codes)
        acknowledged = set(review.acknowledged_warning_codes)
        if not required_warnings.issubset(acknowledged):
            raise CorpusPublicationError(
                "WARNINGS_NOT_ACKNOWLEDGED",
                "Every report warning must be acknowledged before publication.",
            )
        return job

    def _build_source_manifest(
        self,
        job: CorpusStandardizationJob,
        review: CorpusPublicationReview,
        selected: list[CorpusFileManifest],
        created_by: str,
    ) -> SourceManifest:
        roles = {item.role for item in selected}
        artifacts: list[SourceIndexArtifact] = []
        if CorpusFileRole.DOCUMENT in roles:
            artifacts.append(SourceIndexArtifact.DOCUMENT_CHUNKS)
        if roles.intersection({CorpusFileRole.TABLE, CorpusFileRole.LABEL}):
            artifacts.extend(
                [SourceIndexArtifact.DATA_DICTIONARY, SourceIndexArtifact.DATASET_PROFILE]
            )
        if CorpusFileRole.IMAGE in roles:
            artifacts.append(SourceIndexArtifact.IMAGE_TEXT)
        artifacts = list(dict.fromkeys(artifacts))
        if roles.issubset({CorpusFileRole.TABLE, CorpusFileRole.LABEL}):
            mode = SourceIngestionMode.TABULAR_PROFILE_AND_TOOL
        elif roles == {CorpusFileRole.IMAGE}:
            mode = SourceIngestionMode.IMAGE_CORPUS
        elif roles == {CorpusFileRole.DOCUMENT}:
            mode = SourceIngestionMode.DOCUMENT_RAG
        else:
            mode = SourceIngestionMode.MIXED_CURATED_CORPUS
        return SourceManifest(
            source_id=job.metadata.corpus_id,
            manifest_version=job.metadata.snapshot_version,
            status=SourceManifestStatus.APPROVED,
            title=job.metadata.display_name,
            source_type=review.source_type,
            source_url=review.source_url,
            retrieved_at=job.created_at,
            source_hash=job.snapshot_hash,
            hash_scope="Canonical corpus snapshot over relative paths, bytes, and reviewed sidecar.",
            content_origin=review.content_origin,
            license_name=review.license_name,
            license_status=review.license_status,
            redistribution_policy=review.redistribution_policy,
            license_notes=review.license_notes,
            ingestion_policy=SourceIngestionPolicy(
                mode=mode,
                index_artifacts=artifacts,
                analysis_tool_required=bool(
                    roles.intersection({CorpusFileRole.TABLE, CorpusFileRole.LABEL})
                ),
            ),
            expected_assets=SourceExpectedAssets(
                raw_files_min=len(selected),
                documents_min=sum(item.role is CorpusFileRole.DOCUMENT for item in selected),
                images_min=sum(item.role is CorpusFileRole.IMAGE for item in selected),
                tables_min=sum(
                    item.role in {CorpusFileRole.TABLE, CorpusFileRole.LABEL}
                    for item in selected
                ),
                expected_formats=sorted(
                    {item.source_format or item.content_type for item in selected}
                ),
            ),
            dataset_version=job.metadata.snapshot_version,
            access_scope_key=review.access_scope_key,
            created_by=created_by,
            notes=review.review_note,
        )

    def _build_item(
        self,
        job: CorpusStandardizationJob,
        source: CorpusFileManifest,
    ) -> CorpusPublicationItem:
        suffix = hashlib.sha256(source.file_id.encode("utf-8")).hexdigest()[:12]
        base = re.sub(r"[^A-Za-z0-9._-]+", "-", job.metadata.corpus_id).strip("-")
        document_id = f"CORPUS-{base}-{suffix}"[:120]
        revision = re.sub(r"[^A-Za-z0-9._-]+", "-", job.metadata.snapshot_version).strip("-")
        revision = (revision or "snapshot")[:48] + "-" + job.snapshot_hash[:12]
        name = PurePosixPath(source.relative_path).name
        if source.role in {CorpusFileRole.TABLE, CorpusFileRole.LABEL}:
            kind = CorpusPublicationArtifactKind.TABULAR_PROFILE
            document_type = "dataset_profile"
        elif source.role is CorpusFileRole.IMAGE:
            kind = CorpusPublicationArtifactKind.IMAGE_TEXT
            document_type = "image_asset"
        else:
            kind = CorpusPublicationArtifactKind.DOCUMENT
            document_type = "corpus_document"
        return CorpusPublicationItem(
            file_id=source.file_id,
            relative_path=source.relative_path,
            artifact_kind=kind,
            document_id=document_id,
            revision=revision,
            title=source.description or name,
            document_type=document_type,
        )

    def _publication_payload(
        self,
        source: CorpusFileManifest,
        job: CorpusStandardizationJob,
        item: CorpusPublicationItem,
    ) -> tuple[bytes, str, list[dict[str, object]]]:
        if source.standardized_ref is None:
            raise CorpusPublicationError("STANDARDIZED_REF_MISSING", "Standardized artifact missing.")
        images: list[dict[str, object]] = []
        if source.role in {CorpusFileRole.TABLE, CorpusFileRole.LABEL}:
            if source.tabular_profile is None:
                raise CorpusPublicationError("TABULAR_PROFILE_MISSING", "Tabular profile missing.")
            content = self._render_tabular_profile(source, job, source.tabular_profile)
        else:
            content = self.corpus_store.load_object(source.standardized_ref)
        if source.role is CorpusFileRole.IMAGE:
            caption = content.decode("utf-8", errors="replace").strip()
            image_id = f"IMG-{hashlib.sha256(source.file_id.encode()).hexdigest()[:20]}"
            image_content = self.corpus_store.load_object(source.raw_ref)
            image_ref = self.ingestion.store.store_image_asset(
                document_id=item.document_id,
                revision=item.revision,
                image_id=image_id,
                filename=PurePosixPath(source.relative_path).name,
                content=image_content,
                content_type=source.content_type,
                source_hash=source.sha256,
            )
            images.append(
                {
                    "image_id": image_id,
                    "object_ref": image_ref.model_dump(mode="json"),
                    "image_type": "corpus_image",
                    "caption": caption[:4000] or source.description or source.relative_path,
                    "caption_source": "vlm",
                    "caption_confidence": 0.8,
                    "ocr_text": "",
                    "detection_summary": "",
                    "source_page": source.relative_path,
                }
            )
        return content, "text/markdown", images

    @staticmethod
    def _render_tabular_profile(
        source: CorpusFileManifest,
        job: CorpusStandardizationJob,
        profile: TabularDataProfile,
    ) -> bytes:
        lines = [
            f"# Dataset profile: {source.relative_path}",
            "",
            "This governed representation contains schema and bounded aggregate statistics only.",
            "Raw rows are retained privately and are not vectorized.",
            "",
        ]
        related = [
            relation
            for relation in (job.report.relations if job.report else [])
            if source.file_id in {relation.from_file_id, relation.to_file_id}
        ]
        if related:
            lines.extend(["## Declared file relations", ""])
            lines.extend(
                f"- {item.from_file_id} {item.relation_type.value} {item.to_file_id}"
                for item in related
            )
            lines.append("")
        for sheet in profile.sheets:
            lines.extend(
                [
                    f"## Sheet: {sheet.name}",
                    "",
                    f"- Observed rows: {sheet.observed_rows}",
                    f"- Column count: {sheet.column_count}",
                    f"- Sample truncated: {str(sheet.sample_truncated).lower()}",
                    f"- Columns truncated: {str(sheet.columns_truncated).lower()}",
                    "",
                    "### Data dictionary",
                    "",
                ]
            )
            for column in sheet.columns:
                stats = [
                    f"types={json.dumps(column.inferred_types, sort_keys=True)}",
                    f"non_empty={column.non_empty_count}",
                    f"empty={column.empty_count}",
                ]
                if column.numeric_min is not None:
                    stats.extend(
                        [
                            f"min={column.numeric_min}",
                            f"max={column.numeric_max}",
                            f"mean={column.numeric_mean}",
                        ]
                    )
                lines.append(f"- {column.name}: " + ", ".join(stats))
            lines.append("")
        return "\n".join(lines).encode("utf-8")

    def _ingestion_metadata(
        self,
        batch: CorpusPublicationBatch,
        item: CorpusPublicationItem,
        job: CorpusStandardizationJob,
        source: CorpusFileManifest,
        images: list[dict[str, object]],
    ) -> dict[str, object]:
        artifacts = {
            CorpusPublicationArtifactKind.DOCUMENT: [SourceIndexArtifact.DOCUMENT_CHUNKS.value],
            CorpusPublicationArtifactKind.TABULAR_PROFILE: [
                SourceIndexArtifact.DATA_DICTIONARY.value,
                SourceIndexArtifact.DATASET_PROFILE.value,
            ],
            CorpusPublicationArtifactKind.IMAGE_TEXT: [SourceIndexArtifact.IMAGE_TEXT.value],
        }[item.artifact_kind]
        return {
            "document_id": item.document_id,
            "revision": item.revision,
            "title": item.title,
            "document_type": item.document_type,
            "approval_status": ApprovalStatus.APPROVED.value,
            "lifecycle": DocumentLifecycle.PUBLISHED.value,
            "source_kind": job.metadata.source_kind,
            "source_uri": f"{batch.review.source_url}#{source.relative_path}",
            "source_license": batch.review.license_name,
            "source_id": batch.source_manifest.source_id,
            "source_manifest_version": batch.source_manifest.manifest_version,
            "dataset_version": batch.source_manifest.dataset_version,
            "source_license_status": batch.review.license_status.value,
            "redistribution_policy": batch.review.redistribution_policy.value,
            "access_scope_key": batch.review.access_scope_key,
            "retrieval_policy": batch.review.retrieval_policy.value,
            "source_index_artifacts": artifacts,
            "images": images,
        }

    @staticmethod
    def _fingerprint(
        review: CorpusPublicationReview,
        manifest: SourceManifest,
        items: list[CorpusPublicationItem],
    ) -> str:
        canonical = json.dumps(
            {
                "review": review.model_dump(mode="json"),
                "manifest": manifest.model_dump(mode="json", exclude={"created_at"}),
                "items": [
                    item.model_dump(mode="json", exclude={"updated_at"}) for item in items
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _event(
        self,
        batch: CorpusPublicationBatch,
        status: CorpusPublicationStatus,
        message: str,
        progress: int,
    ) -> None:
        batch.status = status
        batch.progress = progress
        batch.events.append(
            CorpusPublicationEvent(
                status=status,
                message=message,
                progress=progress,
                attempt=batch.attempt,
            )
        )
        self.repository.save(batch)

    def _finish_failed(
        self,
        batch: CorpusPublicationBatch,
        code: str,
        message: str,
    ) -> CorpusPublicationBatch:
        batch.status = CorpusPublicationStatus.FAILED
        batch.error_code = code
        batch.safe_error_summary = message
        batch.finished_at = datetime.now(UTC)
        batch.failed_count = sum(
            item.status is CorpusPublicationItemStatus.FAILED for item in batch.items
        )
        batch.published_count = sum(
            item.status is CorpusPublicationItemStatus.PUBLISHED for item in batch.items
        )
        batch.events.append(
            CorpusPublicationEvent(
                status=CorpusPublicationStatus.FAILED,
                message=message,
                progress=batch.progress,
                attempt=batch.attempt,
            )
        )
        return self.repository.save(batch)

    def _require_batch(self, batch_id: str) -> CorpusPublicationBatch:
        batch = self.repository.get(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        return batch
