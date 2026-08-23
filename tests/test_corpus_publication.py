from __future__ import annotations

import hashlib

from semikb.config import Settings
from semikb.contracts.corpus import (
    CorpusFileManifest,
    CorpusFileRole,
    CorpusStandardizationJob,
    CorpusStandardizationMetadata,
    CorpusStandardizationReport,
    CorpusStandardizationStatus,
    TabularColumnProfile,
    TabularDataProfile,
    TabularSheetProfile,
)
from semikb.contracts.corpus_publication import (
    CorpusPublicationItemStatus,
    CorpusPublicationReconciliation,
    CorpusPublicationReview,
    CorpusPublicationStatus,
)
from semikb.contracts.models import (
    RedistributionPolicy,
    SourceContentOrigin,
    SourceLicenseStatus,
    SourceManifestType,
)
from semikb.rag_ingestion.corpus_publication import (
    CorpusPublicationError,
    CorpusPublicationService,
)
from semikb.rag_ingestion.service import IngestionService
from semikb.storage.corpus_publication import DemoCorpusPublicationRepository
from semikb.storage.corpus_standardization import DemoCorpusStandardizationRepository
from semikb.storage.memory import DemoStore


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _service(
    *,
    include_table: bool = False,
    warnings: list[str] | None = None,
) -> tuple[CorpusPublicationService, CorpusStandardizationJob, DemoStore]:
    corpus = DemoCorpusStandardizationRepository()
    ingest_store = DemoStore()
    raw = b"# Chamber clean procedure\n\nVerify pressure and RF match before release."
    raw_ref = corpus.store_raw(
        corpus_id="generic-corpus",
        snapshot_hash=_sha(raw),
        category="uploads",
        relative_path="sop.md",
        content=raw,
        content_type="text/markdown",
    )
    standardized_ref = corpus.store_derived(
        corpus_id="generic-corpus",
        snapshot_hash=_sha(raw),
        category="standardized",
        relative_path="doc/document.md",
        content=raw,
        content_type="text/markdown",
    )
    files = [
        CorpusFileManifest(
            file_id="doc",
            relative_path="sop.md",
            sha256=_sha(raw),
            size_bytes=len(raw),
            content_type="text/markdown",
            role=CorpusFileRole.DOCUMENT,
            source_format="markdown",
            raw_ref=raw_ref,
            standardized_ref=standardized_ref,
            warning_codes=warnings or [],
        )
    ]
    if include_table:
        table = b"pressure,alarm\n1,0\n2,1\n"
        table_ref = corpus.store_raw(
            corpus_id="generic-corpus",
            snapshot_hash=_sha(raw),
            category="uploads",
            relative_path="measurements.csv",
            content=table,
            content_type="text/csv",
        )
        profile = TabularDataProfile(
            sheets=[
                TabularSheetProfile(
                    name="csv",
                    observed_rows=2,
                    column_count=2,
                    columns=[
                        TabularColumnProfile(
                            name="pressure",
                            inferred_types={"number": 2},
                            non_empty_count=2,
                            numeric_min=1,
                            numeric_max=2,
                            numeric_mean=1.5,
                        )
                    ],
                )
            ]
        )
        profile_bytes = profile.model_dump_json().encode()
        profile_ref = corpus.store_derived(
            corpus_id="generic-corpus",
            snapshot_hash=_sha(raw),
            category="standardized",
            relative_path="table/tabular-profile.json",
            content=profile_bytes,
            content_type="application/json",
        )
        files.append(
            CorpusFileManifest(
                file_id="table",
                relative_path="measurements.csv",
                sha256=_sha(table),
                size_bytes=len(table),
                content_type="text/csv",
                role=CorpusFileRole.TABLE,
                source_format="csv",
                raw_ref=table_ref,
                standardized_ref=profile_ref,
                tabular_profile=profile,
            )
        )
    report = CorpusStandardizationReport(
        corpus_id="generic-corpus",
        snapshot_version="v1",
        snapshot_hash=_sha(raw),
        inferred_corpus_kind="mixed" if include_table else "document_collection",
        files=files,
        warning_codes=warnings or [],
    )
    job = CorpusStandardizationJob(
        metadata=CorpusStandardizationMetadata(
            corpus_id="generic-corpus",
            snapshot_version="v1",
            display_name="Generic semiconductor corpus",
        ),
        snapshot_hash=_sha(raw),
        request_fingerprint=_sha(b"request"),
        idempotency_key="generic-corpus:v1",
        status=CorpusStandardizationStatus.REVIEW_REQUIRED,
        report=report,
        created_by="knowledge-admin",
    )
    corpus.create_or_get(job)
    ingestion = IngestionService(
        ingest_store,
        Settings(_env_file=None, demo_mode=True),
    )
    return (
        CorpusPublicationService(
            DemoCorpusPublicationRepository(),
            corpus,
            ingestion,
        ),
        job,
        ingest_store,
    )


def _review(job: CorpusStandardizationJob, selected: list[str]) -> CorpusPublicationReview:
    return CorpusPublicationReview(
        request_id="review-1",
        standardization_job_id=job.job_id,
        expected_snapshot_hash=job.snapshot_hash,
        selected_file_ids=selected,
        acknowledged_warning_codes=job.report.warning_codes if job.report else [],
        source_type=SourceManifestType.CURATED_CORPUS,
        content_origin=SourceContentOrigin.REAL,
        source_url="https://example.com/semiconductor-corpus",
        license_name="Declared interview-demo terms",
        license_status=SourceLicenseStatus.DECLARED,
        redistribution_policy=RedistributionPolicy.RESTRICTED,
        access_scope_key="internal_controlled",
        review_note="Reviewed source scope, bounded representations, and warnings.",
    )


def test_publication_reuses_governed_ingestion_and_reconciles_all_stores() -> None:
    service, job, store = _service(include_table=True)

    submitted = service.submit(_review(job, ["doc", "table"]), created_by="admin")
    completed = service.process(submitted.batch_id)

    assert completed.status is CorpusPublicationStatus.COMPLETED
    assert completed.published_count == 2
    assert all(item.status is CorpusPublicationItemStatus.PUBLISHED for item in completed.items)
    assert all(item.reconciliation and item.reconciliation.passed for item in completed.items)
    assert all(document.lifecycle.value == "published" for document in store.documents.values())
    table_document = next(
        document for document in store.documents.values() if document.document_type == "dataset_profile"
    )
    table_chunks = [
        chunk.chunk_text
        for chunk in store.chunks.values()
        if chunk.document_id == table_document.document_id
    ]
    assert any("Raw rows are retained privately" in text for text in table_chunks)
    assert any("Observed preview rows: 2" in text for text in table_chunks)
    assert any("Full row count: 2" in text for text in table_chunks)
    assert not any("1,0" in text or "2,1" in text for text in table_chunks)


def test_publication_is_idempotent_and_requires_warning_acknowledgement() -> None:
    service, job, _ = _service(warnings=["CORPUS_REVIEW_SAMPLE_TRUNCATED"])
    review = _review(job, ["doc"])
    first = service.submit(review, created_by="admin")
    second = service.submit(review, created_by="admin")
    assert second.batch_id == first.batch_id

    unacknowledged = review.model_copy(
        update={"request_id": "review-2", "acknowledged_warning_codes": []}
    )
    try:
        service.submit(unacknowledged, created_by="admin")
    except CorpusPublicationError as exc:
        assert exc.code == "WARNINGS_NOT_ACKNOWLEDGED"
    else:
        raise AssertionError("Warnings must be acknowledged before publication.")


def test_reconciliation_failure_is_quarantined_and_retry_uses_new_revision(
    monkeypatch,
) -> None:
    service, job, store = _service()
    original_reconcile = store.reconcile_published_document
    calls = 0

    def fail_once(document_id: str, revision: str) -> CorpusPublicationReconciliation:
        nonlocal calls
        calls += 1
        if calls == 1:
            return CorpusPublicationReconciliation(
                document_count=1,
                chunk_count=1,
                vector_count=0,
                passed=False,
                warning_codes=["VECTOR_READBACK_MISMATCH"],
            )
        return original_reconcile(document_id, revision)

    monkeypatch.setattr(store, "reconcile_published_document", fail_once)
    submitted = service.submit(_review(job, ["doc"]), created_by="admin")
    failed = service.process(submitted.batch_id)

    assert failed.status is CorpusPublicationStatus.FAILED
    assert failed.items[0].error_code == "CROSS_STORE_RECONCILIATION_FAILED"
    assert all(chunk.lifecycle.value != "published" for chunk in store.chunks.values())
    first_revision = failed.items[0].revision

    retry = service.prepare_retry(failed.batch_id)
    assert retry.items[0].revision != first_revision
    completed = service.process(retry.batch_id)

    assert completed.status is CorpusPublicationStatus.COMPLETED
    assert completed.items[0].reconciliation is not None
    assert completed.items[0].reconciliation.passed is True
