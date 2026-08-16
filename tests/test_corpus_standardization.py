from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass

import pytest
from PIL import Image
from pydantic import ValidationError

from semikb.config import Settings
from semikb.contracts.corpus import (
    CorpusFileRole,
    CorpusSidecar,
    CorpusStandardizationError,
    CorpusStandardizationJob,
    CorpusStandardizationMetadata,
    CorpusStandardizationStatus,
    CorpusUploadedFile,
)
from semikb.rag_ingestion.corpus_standardization import CorpusStandardizationService
from semikb.rag_ingestion.semikb_adapter import SemikbIngestAdapter
from semikb.storage.corpus_standardization import (
    CorpusStandardizationConflictError,
    DemoCorpusStandardizationRepository,
)
from semikb_ingest.providers import ProviderRegistry, VisionAnalysis


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    return output.getvalue()


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (24, 24), (20, 160, 120)).save(output, "PNG")
    return output.getvalue()


@dataclass
class FakeVisionProvider:
    provider_name: str = "qwen-vl"
    provider_version: str = "qwen3.7-plus-test"

    def analyze_image(self, **_kwargs) -> VisionAnalysis:
        return VisionAnalysis(
            caption="图中可见规则网格和一处颜色差异。",
            ocr_text="",
            detection_summary="仅描述可见像素，不推断设备、配方或根因。",
            confidence=0.91,
            detected_language="zh",
        )


def _service() -> tuple[CorpusStandardizationService, DemoCorpusStandardizationRepository]:
    settings = Settings(_env_file=None, demo_mode=True)
    registry = ProviderRegistry()
    registry.register(FakeVisionProvider())
    store = DemoCorpusStandardizationRepository()
    adapter = SemikbIngestAdapter(settings, registry)
    return CorpusStandardizationService(store, settings, adapter), store


def _metadata(corpus_id: str = "generic-corpus") -> CorpusStandardizationMetadata:
    return CorpusStandardizationMetadata(
        corpus_id=corpus_id,
        snapshot_version="v1",
        display_name="Unknown future upload",
        source_kind="user_upload",
        source_license="unknown",
        corpus_kind="auto",
    )


def test_mixed_archive_uses_generic_dispatch_profile_image_text_and_explicit_relation() -> None:
    service, store = _service()
    archive = _zip_bytes(
        {
            "docs/guide.md": b"# Guide\n\nInspect only the observed signal.\n",
            "tables/measurements.csv": b"wafer,value\nW-001,1.2\nW-002,1.5\n",
            "labels/annotations.csv": b"image,label\nwafer.png,edge\n",
            "images/wafer.png": _png_bytes(),
            "notes/raw.json": b'{"retained": true}',
        }
    )
    sidecar = CorpusSidecar.model_validate(
        {
            "files": [
                {"path": "snapshot/labels/annotations.csv", "role": "label"},
            ],
            "relations": [
                {
                    "from_glob": "snapshot/labels/*.csv",
                    "to_glob": "snapshot/images/*.png",
                    "relation_type": "labels",
                }
            ],
        }
    )
    submitted = service.submit(
        [
            CorpusUploadedFile(
                relative_path="snapshot.zip",
                content_type="application/zip",
                content=archive,
            )
        ],
        _metadata(),
        sidecar,
        "tester",
    )

    completed = service.process(submitted.job_id)

    assert completed.status is CorpusStandardizationStatus.REVIEW_REQUIRED
    assert completed.report is not None
    by_path = {item.relative_path: item for item in completed.report.files}
    assert by_path["snapshot/docs/guide.md"].role is CorpusFileRole.DOCUMENT
    assert by_path["snapshot/tables/measurements.csv"].role is CorpusFileRole.TABLE
    assert by_path["snapshot/labels/annotations.csv"].role is CorpusFileRole.LABEL
    assert by_path["snapshot/images/wafer.png"].role is CorpusFileRole.IMAGE
    assert by_path["snapshot/notes/raw.json"].role is CorpusFileRole.UNSUPPORTED
    assert len(completed.report.relations) == 1
    assert completed.report.relations[0].relation_type.value == "labels"
    assert by_path["snapshot/images/wafer.png"].standardized_ref is not None
    image_text = store.load_object(by_path["snapshot/images/wafer.png"].standardized_ref).decode()
    assert "规则网格" in image_text
    assert "根因是" not in image_text
    table_ref = by_path["snapshot/tables/measurements.csv"].standardized_ref
    assert table_ref is not None
    table_profile = store.load_object(table_ref).decode()
    assert "W-001" not in table_profile
    assert '"raw_rows_vectorized": false' in table_profile


def test_renaming_source_and_corpus_does_not_change_route_behavior() -> None:
    service, _ = _service()
    payload = b"signal,value\npressure,12\n"
    results = []
    for corpus_id, filename in (("alpha", "first.csv"), ("renamed", "unseen.csv")):
        job = service.submit(
            [CorpusUploadedFile(relative_path=filename, content_type="text/csv", content=payload)],
            _metadata(corpus_id),
            None,
            "tester",
        )
        result = service.process(job.job_id)
        assert result.report is not None
        file = next(item for item in result.report.files if item.role is not CorpusFileRole.ARCHIVE)
        results.append((file.role, file.source_format, file.parser_name, result.report.inferred_corpus_kind))
    assert results[0] == results[1]


def test_sidecar_can_declare_headerless_whitespace_table_without_source_special_case() -> None:
    service, store = _service()
    job = service.submit(
        [
            CorpusUploadedFile(
                relative_path="future.data",
                content_type="application/octet-stream",
                content=b"1.0 NaN 3.0\n2.0 4.0 6.0\n",
            )
        ],
        _metadata("delimited-generic"),
        CorpusSidecar.model_validate(
            {
                "files": [
                    {
                        "path": "future.data",
                        "role": "table",
                        "tabular_delimiter": "whitespace",
                        "tabular_has_header": False,
                    }
                ]
            }
        ),
        "tester",
    )
    completed = service.process(job.job_id)
    assert completed.status is CorpusStandardizationStatus.REVIEW_REQUIRED
    assert completed.report is not None
    manifest = completed.report.files[0]
    assert manifest.source_format == "delimited_text"
    assert manifest.tabular_profile is not None
    assert [item.name for item in manifest.tabular_profile.sheets[0].columns] == [
        "column_1",
        "column_2",
        "column_3",
    ]
    assert manifest.standardized_ref is not None
    assert b"1.0 NaN 3.0" not in store.load_object(manifest.standardized_ref)


@pytest.mark.parametrize(
    ("archive_files", "error_code"),
    [
        ({"../escape.md": b"bad"}, "CORPUS_UNSAFE_PATH"),
        ({"nested.zip": _zip_bytes({"note.md": b"# Nested"})}, "CORPUS_NESTED_ARCHIVE_REJECTED"),
        ({"run.ps1": b"Write-Host bad"}, "CORPUS_DANGEROUS_FILE_REJECTED"),
    ],
)
def test_unsafe_archive_fails_closed_without_review_report(
    archive_files: dict[str, bytes],
    error_code: str,
) -> None:
    service, _ = _service()
    job = service.submit(
        [
            CorpusUploadedFile(
                relative_path="unsafe.zip",
                content_type="application/zip",
                content=_zip_bytes(archive_files),
            )
        ],
        _metadata(),
        None,
        "tester",
    )
    failed = service.process(job.job_id)
    assert failed.status is CorpusStandardizationStatus.FAILED
    assert failed.error_code == error_code
    assert failed.report is None


def test_idempotency_replays_equal_request_and_rejects_changed_metadata() -> None:
    service, _ = _service()
    upload = CorpusUploadedFile(
        relative_path="note.md",
        content_type="text/markdown",
        content=b"# Generic note",
    )
    first = service.submit([upload], _metadata(), None, "tester")
    replay = service.submit([upload], _metadata(), None, "tester")
    assert replay.job_id == first.job_id
    changed = _metadata().model_copy(update={"source_license": "different"})
    with pytest.raises(CorpusStandardizationConflictError):
        service.submit([upload], changed, None, "tester")


def test_unknown_sidecar_fields_and_missing_relation_targets_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CorpusSidecar.model_validate({"unknown": True})
    service, _ = _service()
    job = service.submit(
        [
            CorpusUploadedFile(
                relative_path="note.md",
                content_type="text/markdown",
                content=b"# Note",
            )
        ],
        _metadata(),
        CorpusSidecar.model_validate(
            {
                "relations": [
                    {
                        "from_glob": "missing/*.csv",
                        "to_glob": "*.md",
                        "relation_type": "labels",
                    }
                ]
            }
        ),
        "tester",
    )
    failed = service.process(job.job_id)
    assert failed.error_code == "CORPUS_RELATION_TARGET_NOT_FOUND"


def test_direct_dangerous_upload_is_rejected_before_snapshot_job() -> None:
    service, _ = _service()
    with pytest.raises(CorpusStandardizationError, match="Executable"):
        service.submit(
            [
                CorpusUploadedFile(
                    relative_path="run.cmd",
                    content_type="application/octet-stream",
                    content=b"echo bad",
                )
            ],
            _metadata(),
            None,
            "tester",
        )


def test_report_json_is_reviewable_and_contains_no_raw_file_content() -> None:
    service, store = _service()
    job = service.submit(
        [
            CorpusUploadedFile(
                relative_path="unknown.json",
                content_type="application/json",
                content=json.dumps({"secret_row": "not-in-report"}).encode(),
            )
        ],
        _metadata(),
        None,
        "tester",
    )
    completed = service.process(job.job_id)
    assert completed.report_ref is not None
    report = store.load_object(completed.report_ref).decode()
    assert "CORPUS_UNSUPPORTED_FILE_RETAINED" in report
    assert "not-in-report" not in report


def test_job_creator_has_no_demo_default() -> None:
    assert CorpusStandardizationJob.model_fields["created_by"].is_required()


def test_transient_storage_failure_uses_retryable_safe_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, store = _service()
    job = service.submit(
        [
            CorpusUploadedFile(
                relative_path="note.md",
                content_type="text/markdown",
                content=b"# Generic note",
            )
        ],
        _metadata("transient-failure"),
        None,
        "tester",
    )

    def timeout(_object_ref: object) -> bytes:
        raise TimeoutError("provider detail must not be persisted")

    monkeypatch.setattr(store, "load_object", timeout)
    failed = service.process(job.job_id)

    assert failed.status is CorpusStandardizationStatus.FAILED
    assert failed.error_code == "TIMEOUTERROR"
    assert failed.safe_error_summary == (
        "Corpus standardization failed safely. No knowledge data was published."
    )
    assert "provider detail" not in failed.model_dump_json()
