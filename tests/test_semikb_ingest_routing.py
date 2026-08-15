from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from semikb_ingest.errors import ERROR_CATALOG, IngestError, IngestErrorCode
from semikb_ingest.models import SourceFormat
from semikb_ingest.routing import FormatRouter, RoutingPolicy, route_table


def office_payload(family: SourceFormat, *extra_names: str, body: bytes = b"<xml/>") -> bytes:
    family_member = {
        SourceFormat.DOCX: "word/document.xml",
        SourceFormat.XLSX: "xl/workbook.xml",
        SourceFormat.PPTX: "ppt/presentation.xml",
    }[family]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"<Types/>")
        archive.writestr(family_member, body)
        for name in extra_names:
            archive.writestr(name, b"unsafe")
    return output.getvalue()


@pytest.mark.parametrize(
    ("filename", "media_type", "content", "expected"),
    [
        ("sop.md", "text/markdown", b"# SOP", SourceFormat.MARKDOWN),
        ("notes.txt", "text/plain; charset=utf-8", b"notes", SourceFormat.TEXT),
        ("manual.html", "text/html", b"<h1>Manual</h1>", SourceFormat.HTML),
        ("alarm.pdf", "application/pdf", b"%PDF-1.7\n", SourceFormat.PDF),
        (
            "recipe.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            office_payload(SourceFormat.DOCX),
            SourceFormat.DOCX,
        ),
        (
            "spc.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            office_payload(SourceFormat.XLSX),
            SourceFormat.XLSX,
        ),
        ("measurements.csv", "text/csv", b"wafer,cd\nW01,32", SourceFormat.CSV),
        (
            "review.pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            office_payload(SourceFormat.PPTX),
            SourceFormat.PPTX,
        ),
        ("wafer.png", "image/png", b"\x89PNG\r\n\x1a\nbody", SourceFormat.IMAGE),
        ("defect.jpeg", "image/jpeg", b"\xff\xd8\xffbody", SourceFormat.IMAGE),
    ],
)
def test_supported_formats_use_explicit_routes(
    filename: str,
    media_type: str,
    content: bytes,
    expected: SourceFormat,
) -> None:
    resolved = FormatRouter().resolve(filename, content, media_type)

    assert resolved.source_format is expected
    assert resolved.parser_id.endswith("-v1")


def test_route_table_matches_frozen_capability_matrix() -> None:
    matrix = json.loads(
        Path("docs/evidence/t9-4-4-1/format-capability-matrix-v1.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = {entry["format"]: entry for entry in matrix["formats"]}
    active = {route.source_format.value: route for route in route_table()}

    assert set(active) == set(frozen)
    for source_format, route in active.items():
        expected = frozen[source_format]
        assert route.parser_id == expected["parser_id"]
        assert route.provider == expected["provider"]
        assert route.fallback == expected["fallback"]
        assert {rule.extension for rule in route.rules} == set(expected["extensions"])

    assert [route.source_format for route in route_table() if route.provider == "mineru"] == [
        SourceFormat.PDF
    ]
    assert all(route.fallback != "mineru" for route in route_table())


def test_error_catalog_matches_frozen_error_artifact() -> None:
    artifact = json.loads(
        Path("docs/evidence/t9-4-4-1/semikb-ingest-errors-v1.json").read_text(
            encoding="utf-8"
        )
    )
    frozen = {entry["code"]: entry for entry in artifact["errors"]}

    assert set(frozen) == {code.value for code in ERROR_CATALOG}
    for code, descriptor in ERROR_CATALOG.items():
        assert descriptor.http_status == frozen[code.value]["http_status"]
        assert descriptor.stage.value == frozen[code.value]["stage"]
        assert descriptor.retry_policy.value == frozen[code.value]["retry_policy"]
        assert descriptor.quarantine is frozen[code.value]["quarantine"]


@pytest.mark.parametrize(
    ("filename", "media_type", "content", "expected_code"),
    [
        ("payload.json", "application/json", b"{}", IngestErrorCode.UNSUPPORTED_FORMAT),
        ("manual.pdf", "image/png", b"%PDF-1.7", IngestErrorCode.FILE_TYPE_MISMATCH),
        (
            "manual.pdf",
            "application/pdf",
            b"\x89PNG\r\n\x1a\nbody",
            IngestErrorCode.FILE_TYPE_MISMATCH,
        ),
        (
            "wafer.png",
            "image/png",
            b"\xff\xd8\xffbody",
            IngestErrorCode.FILE_TYPE_MISMATCH,
        ),
        (
            "recipe.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            office_payload(SourceFormat.XLSX),
            IngestErrorCode.FILE_TYPE_MISMATCH,
        ),
        (
            "recipe.docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            b"PK\x03\x04not-a-zip",
            IngestErrorCode.INVALID_OFFICE_CONTAINER,
        ),
    ],
)
def test_spoofed_or_unsupported_files_fail_with_stable_codes(
    filename: str,
    media_type: str,
    content: bytes,
    expected_code: IngestErrorCode,
) -> None:
    with pytest.raises(IngestError) as captured:
        FormatRouter().resolve(filename, content, media_type)

    assert captured.value.code is expected_code


def test_extensionless_binary_requires_compatible_signals() -> None:
    resolved = FormatRouter().resolve("upload", b"%PDF-1.7\n", "application/pdf")
    assert resolved.source_format is SourceFormat.PDF

    office = FormatRouter().resolve(
        "upload",
        office_payload(SourceFormat.DOCX),
        "application/zip",
    )
    assert office.source_format is SourceFormat.DOCX

    with pytest.raises(IngestError) as captured:
        FormatRouter().resolve("upload", b"plain text", "text/plain")
    assert captured.value.code is IngestErrorCode.UNSUPPORTED_FORMAT

    with pytest.raises(IngestError) as mismatch:
        FormatRouter().resolve(
            "upload",
            b"not-an-office-container",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    assert mismatch.value.code is IngestErrorCode.FILE_TYPE_MISMATCH


def test_office_archive_limits_and_paths_are_enforced() -> None:
    compressed = office_payload(SourceFormat.DOCX, body=b"A" * 20_000)
    policy = RoutingPolicy(max_office_compression_ratio=2)
    with pytest.raises(IngestError) as compressed_error:
        FormatRouter(policy).resolve("recipe.docx", compressed)
    assert compressed_error.value.code is IngestErrorCode.ZIP_BOMB_SUSPECTED
    assert compressed_error.value.descriptor.quarantine is True

    traversal = office_payload(SourceFormat.DOCX, "../escape.bin")
    with pytest.raises(IngestError) as traversal_error:
        FormatRouter().resolve("recipe.docx", traversal)
    assert traversal_error.value.code is IngestErrorCode.INVALID_OFFICE_CONTAINER


def test_source_size_limit_is_checked_before_routing() -> None:
    router = FormatRouter(RoutingPolicy(max_source_bytes=4))
    with pytest.raises(IngestError) as captured:
        router.resolve("sop.md", b"12345", "text/markdown")
    assert captured.value.code is IngestErrorCode.DOCUMENT_LIMIT_EXCEEDED
