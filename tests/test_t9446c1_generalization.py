from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.export_t9446c1_contracts import DEFAULT_OUTPUT, export_contracts
from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    ApprovalStatus,
    Chunk,
    DocumentLifecycle,
    EvaluationDatasetPurpose,
    EvaluationLeakageStatus,
    IngestionStatus,
    IngestUploadMetadata,
    RetrievalConstraints,
    RetrievalMode,
    RetrievalPolicy,
)
from semikb.demo_factory import load_demo_source_manifest
from semikb.evaluation.service import EvaluationService
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.production_service import (
    ProductionRetrievalService,
    RetrievalOptions,
)
from semikb.rag_retrieval.service import RetrievalService, tokenize
from semikb.storage.memory import DemoStore


def _manifest(source_id: str = "renamed.public.source"):
    original = load_demo_source_manifest(
        Path("data/source_manifests/semikb-demo-corpus-v1.json")
    )
    return original.model_copy(update={"source_id": source_id})


def _published_payload(document_id: str, source_id: str) -> dict[str, object]:
    return {
        "document_id": document_id,
        "revision": "R1",
        "title": "Generic governed procedure",
        "document_type": "sop",
        "content": "# Procedure\n\nVerify the observed condition before changing settings.",
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": "public_document",
        "source_license": "CC0-1.0",
        "source_id": source_id,
        "source_manifest_version": "1.0.0",
        "dataset_version": "demo-v2",
        "source_license_status": "verified",
        "redistribution_policy": "allowed",
        "access_scope_key": "demo_engineering",
        "retrieval_policy": "protected",
    }


def test_public_contract_defaults_are_neutral_and_fail_closed() -> None:
    actor = ActorScope()
    upload = IngestUploadMetadata(
        document_id="UNKNOWN-001",
        revision="R1",
        title="Unknown upload",
        document_type="unknown",
    )

    assert actor.user_id == "anonymous"
    assert not actor.roles
    assert not actor.access_scope_keys
    assert upload.approval_status is ApprovalStatus.DRAFT
    assert upload.lifecycle is DocumentLifecycle.STAGED
    assert upload.access_scope_key is None
    assert upload.fab is None
    assert upload.product is None
    assert upload.source_license == "unknown"

    with pytest.raises(ValidationError, match="reviewed governance fields"):
        IngestUploadMetadata(
            document_id="UNKNOWN-002",
            revision="R1",
            title="Unsafe publication",
            document_type="unknown",
            approval_status="approved",
            lifecycle="published",
        )


def test_source_and_document_renaming_do_not_change_publication_behavior() -> None:
    store = DemoStore()
    manifest = _manifest("arbitrary.snapshot.name")
    store.register_source_manifest(manifest)
    service = IngestionService(
        store,
        Settings(_env_file=None, demo_mode=True, embedding_dim=8),
    )

    job = service.ingest_payload(
        _published_payload("UNSEEN-DOCUMENT-NAME", manifest.source_id)
    )

    assert job.status is IngestionStatus.PUBLISHED
    document = store.get_document("UNSEEN-DOCUMENT-NAME", "R1")
    assert document is not None
    assert document.retrieval_policy is RetrievalPolicy.PROTECTED


def test_missing_source_manifest_keeps_requested_publication_out_of_index() -> None:
    store = DemoStore()
    service = IngestionService(
        store,
        Settings(_env_file=None, demo_mode=True, embedding_dim=8),
    )

    job = service.ingest_payload(
        _published_payload("UNREVIEWED-DOCUMENT", "missing.snapshot")
    )

    assert job.status is IngestionStatus.FAILED
    assert job.error_code == "PUBLICATION_GATE_SOURCE_MANIFEST_MISSING"
    assert store.get_document("UNREVIEWED-DOCUMENT", "R1") is None


def test_protected_evidence_uses_governed_policy_not_document_prefix() -> None:
    protected = Chunk(
        chunk_id="C-PROTECTED",
        document_id="RENAMED-GUIDE",
        revision="R1",
        chunk_text="controlled procedure",
        page_or_section="section",
        approval_status="approved",
        lifecycle="published",
        tool_id="ETCH-03",
        chamber="B",
        retrieval_policy="protected",
    )
    misleading_prefix = protected.model_copy(
        update={
            "chunk_id": "C-STANDARD",
            "document_id": "SOP-MISLEADING",
            "retrieval_policy": RetrievalPolicy.STANDARD,
        }
    )

    assert ProductionRetrievalService._is_protected_evidence(
        protected,
        "Review ETCH-03 Chamber B",
    )
    assert not ProductionRetrievalService._is_protected_evidence(
        misleading_prefix,
        "Review ETCH-03 Chamber B",
    )


def test_hyde_and_image_boost_use_structured_modes_for_unseen_phrasing() -> None:
    service = object.__new__(ProductionRetrievalService)
    service.settings = Settings(_env_file=None, demo_mode=True, hyde_enabled=True)
    options = RetrievalOptions()

    assert service._should_use_hyde(
        "Provide a cross-domain causal assessment using unfamiliar wording.",
        RetrievalConstraints(retrieval_mode=RetrievalMode.DIAGNOSTIC),
        options,
    )
    assert not service._should_use_hyde(
        "异常原因怎么排查",
        RetrievalConstraints(retrieval_mode=RetrievalMode.STANDARD),
        options,
    )

    image_chunk = Chunk(
        chunk_id="IMAGE-1",
        document_id="VISUAL-EVIDENCE",
        revision="R1",
        chunk_type="image_text",
        chunk_text="spatial pattern evidence",
        page_or_section="image",
    )
    query_tokens = tokenize("show spatial evidence")
    standard = RetrievalService._score(image_chunk, query_tokens, RetrievalMode.STANDARD)
    image = RetrievalService._score(image_chunk, query_tokens, RetrievalMode.IMAGE)
    assert image[1] == pytest.approx(standard[1] + 0.08)


def test_holdout_dataset_is_sealed_audited_and_opened_once(tmp_path: Path) -> None:
    payload = {
        "dataset_version": "holdout-v1",
        "source_kind": "synthetic",
        "description": "Sealed unseen questions",
        "purpose": "holdout",
        "sealed_at": "2026-08-16T00:00:00Z",
        "source_snapshot_hash": "a" * 64,
        "leakage_status": "cleared",
        "cases": [
            {
                "case_id": "blind-001",
                "question": "Unseen question",
                "expected_chunk_ids": [],
                "expected_outcome": "no_evidence",
            }
        ],
    }
    (tmp_path / "holdout_v1.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    store = DemoStore()
    evaluation = EvaluationService(store, object(), tmp_path)

    first = evaluation.create_run("holdout-v1")
    second = evaluation.create_run("holdout-v1")

    assert first.dataset_purpose is EvaluationDatasetPurpose.HOLDOUT
    assert first.dataset_leakage_status is EvaluationLeakageStatus.CLEARED
    assert first.dataset_opened_at is not None
    assert second.dataset_opened_at == first.dataset_opened_at
    assert first.source_snapshot_hash == "a" * 64


def test_invalid_profile_does_not_open_holdout_dataset(tmp_path: Path) -> None:
    payload = {
        "dataset_version": "holdout-invalid-profile-v1",
        "source_kind": "synthetic",
        "description": "Sealed unseen questions",
        "purpose": "holdout",
        "sealed_at": "2026-08-16T00:00:00Z",
        "source_snapshot_hash": "b" * 64,
        "leakage_status": "cleared",
        "cases": [
            {
                "case_id": "blind-invalid-profile-001",
                "question": "Unseen question",
                "expected_chunk_ids": [],
                "expected_outcome": "no_evidence",
            }
        ],
    }
    (tmp_path / "holdout_invalid_profile_v1.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    store = DemoStore()
    evaluation = EvaluationService(store, object(), tmp_path)

    with pytest.raises(ValueError, match="Unsupported retrieval profile"):
        evaluation.create_run(
            "holdout-invalid-profile-v1",
            retrieval_profile="not-a-profile",
        )

    assert store.get_evaluation_dataset("holdout-invalid-profile-v1") is None


def test_legacy_regression_hash_and_versioned_contract_remain_stable() -> None:
    evaluation = EvaluationService(
        DemoStore(),
        object(),
        Path("data/golden_sets"),
    )
    dataset = evaluation._load_dataset("demo-v1")

    assert dataset.dataset_hash == (
        "9f17e60d5082304b9e537c6907cba5e80d55a4d14b90a76598f624b7bead0512"
    )
    assert dataset.purpose is EvaluationDatasetPurpose.REGRESSION
    assert export_contracts(DEFAULT_OUTPUT, check=True)["status"] == "current"
