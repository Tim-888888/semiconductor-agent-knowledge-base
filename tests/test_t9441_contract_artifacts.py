from __future__ import annotations

import json
import re
from pathlib import Path

ARTIFACT_ROOT = Path("docs/evidence/t9-4-4-1")


def _load_json(filename: str) -> dict[str, object]:
    return json.loads((ARTIFACT_ROOT / filename).read_text(encoding="utf-8"))


def test_knowhere_source_lock_is_exact_and_non_vendoring() -> None:
    source_lock = _load_json("knowhere-source-lock.json")

    assert source_lock["repository"] == "https://github.com/Ontos-AI/knowhere.git"
    assert source_lock["commit"] == "2e4eb5846249d273b11902ee00f26db949e45b38"
    assert source_lock["license"]["spdx"] == "Apache-2.0"
    assert re.fullmatch(
        r"[0-9a-f]{64}", source_lock["license"]["license_file_sha256"]
    )
    assert re.fullmatch(
        r"[0-9a-f]{64}", source_lock["license"]["notice_file_sha256"]
    )
    assert source_lock["reuse_status"] == "audit_only_no_vendored_source"
    assert source_lock["vendored_files"] == []
    assert len(source_lock["audited_blobs"]) == 17
    assert all(
        re.fullmatch(r"[0-9a-f]{40}", blob)
        for blob in source_lock["audited_blobs"].values()
    )


def test_format_matrix_has_explicit_routes_without_universal_mineru_fallback() -> None:
    matrix = _load_json("format-capability-matrix-v1.json")
    formats = matrix["formats"]
    by_format = {entry["format"]: entry for entry in formats}

    assert set(by_format) == {
        "markdown",
        "text",
        "html",
        "pdf",
        "docx",
        "xlsx",
        "csv",
        "pptx",
        "image",
    }
    assert matrix["default_policy"]["unknown_format"] == "reject"
    assert matrix["default_policy"]["provider_fallback"] == "format_scoped_only"
    assert by_format["pdf"]["provider"] == "mineru"
    assert all(
        entry["provider"] != "mineru"
        for entry in formats
        if entry["format"] != "pdf"
    )
    assert all(entry["fallback"] != "mineru" for entry in formats)
    assert ".doc" in matrix["explicitly_out_of_scope"]
    assert ".xls" in matrix["explicitly_out_of_scope"]


def test_ingest_schema_freezes_four_domain_outputs_and_provenance() -> None:
    schema = _load_json("semikb-ingest-contract-v1.schema.json")
    definitions = schema["$defs"]

    assert schema["title"] == "ParsedDocument"
    assert schema["additionalProperties"] is False
    assert {
        "ChunkDraft",
        "ImageAssetDraft",
        "TableAssetDraft",
        "ParseProvenance",
    }.issubset(definitions)
    provenance_required = set(definitions["ParseProvenance"]["required"])
    assert {
        "parser_name",
        "parser_version",
        "provider_name",
        "provider_version",
        "upstream_project",
        "upstream_commit",
        "source_sha256",
        "detected_format",
    }.issubset(provenance_required)
    assert definitions["BinaryPayload"]["properties"]["handle"]["minLength"] == 1


def test_ingest_error_codes_are_stable_unique_and_stage_scoped() -> None:
    catalog = _load_json("semikb-ingest-errors-v1.json")
    errors = catalog["errors"]
    codes = [entry["code"] for entry in errors]

    assert len(codes) == len(set(codes))
    assert all(re.fullmatch(r"INGEST_[A-Z0-9_]+", code) for code in codes)
    assert {
        "INGEST_UNSUPPORTED_FORMAT",
        "INGEST_FILE_TYPE_MISMATCH",
        "INGEST_INVALID_OFFICE_CONTAINER",
        "INGEST_PARSER_UNAVAILABLE",
        "INGEST_PARSER_TIMEOUT",
        "INGEST_QUALITY_GATE_FAILED",
        "INGEST_CONTRACT_VIOLATION",
    }.issubset(codes)
    assert {entry["stage"] for entry in errors}.issubset(
        {"validating", "parsing", "quality_check"}
    )
