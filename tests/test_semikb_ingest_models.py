from __future__ import annotations

import ast
import hashlib
import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from semikb_ingest.assets import ProcessPayloadStore
from semikb_ingest.models import (
    ChunkDraft,
    ChunkType,
    ParsedDocument,
    ParseMetrics,
    ParseProvenance,
    SourceFormat,
)


def provenance() -> ParseProvenance:
    return ParseProvenance(
        parser_name="markdown-structured-v1",
        parser_version="1.0.0",
        provider_name=None,
        provider_version=None,
        upstream_project="Ontos-AI/knowhere",
        upstream_commit="2e4eb5846249d273b11902ee00f26db949e45b38",
        source_filename="sop.md",
        source_media_type="text/markdown",
        source_sha256="0" * 64,
        detected_format="markdown",
    )


def test_parsed_document_emits_the_frozen_public_contract() -> None:
    document = ParsedDocument(
        source_format=SourceFormat.MARKDOWN,
        normalized_markdown="# Alarm handling",
        chunks=(
            ChunkDraft(
                draft_id="draft_sop-001",
                chunk_type=ChunkType.TEXT,
                text="Verify chamber pressure.",
                title_path=("Alarm handling",),
            ),
        ),
        provenance=provenance(),
        metrics=ParseMetrics(chunks=1),
    )
    payload = document.model_dump(mode="json")
    schema = json.loads(
        Path("docs/evidence/t9-4-4-1/semikb-ingest-contract-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["contract_version"] == "semikb-ingest-v1"
    assert payload["source_format"] == "markdown"
    assert set(schema["required"]).issubset(payload)
    assert payload["chunks"][0]["location"]["section_path"] == []


def test_public_contract_rejects_unknown_fields_and_omits_governance() -> None:
    with pytest.raises(ValidationError):
        ParsedDocument(
            source_format=SourceFormat.TEXT,
            normalized_markdown="text",
            provenance=provenance(),
            document_id="must-stay-in-rag-ingestion",
        )

    governed_fields = {
        "document_id",
        "revision",
        "approval_status",
        "effective_at",
        "fab",
        "product",
        "tool_id",
        "chamber",
        "recipe_id",
        "access_scope_key",
        "index_version",
    }
    assert governed_fields.isdisjoint(ParsedDocument.model_fields)


def test_public_contract_rejects_duplicate_and_dangling_references() -> None:
    with pytest.raises(ValidationError, match="Asset references must be unique"):
        ChunkDraft(
            draft_id="draft_duplicate",
            chunk_type=ChunkType.TEXT,
            text="duplicate",
            image_asset_ids=("image-1", "image-1"),
        )

    with pytest.raises(ValidationError, match="does not exist"):
        ParsedDocument(
            source_format=SourceFormat.TEXT,
            normalized_markdown="dangling",
            chunks=(
                ChunkDraft(
                    draft_id="draft_dangling",
                    chunk_type=ChunkType.TEXT,
                    text="dangling",
                    image_asset_ids=("missing-image",),
                ),
            ),
            provenance=provenance(),
        )


def test_process_payload_handle_is_verified_and_consumed_once() -> None:
    store = ProcessPayloadStore()
    content = b"synthetic-image-bytes"
    payload = store.put("wafer.png", "image/png", content)

    assert payload.handle.startswith("payload_")
    assert payload.sha256 == hashlib.sha256(content).hexdigest()
    assert store.read(payload) == content
    assert store.pop(payload) == content
    with pytest.raises(KeyError):
        store.read(payload)


def test_package_is_built_and_does_not_import_business_modules() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    packages = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/semikb_ingest" in packages

    violations: list[str] = []
    for source_path in Path("src/semikb_ingest").rglob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "semikb" or name.startswith("semikb.") for name in names):
                violations.append(str(source_path))
    assert violations == []
