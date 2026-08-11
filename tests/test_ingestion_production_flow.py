from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

from semikb.config import Settings
from semikb.contracts.models import DocumentLifecycle, IngestionStatus
from semikb.rag_ingestion.mineru import MinerUPrecisionClient
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.encoders import (
    DeterministicHybridEncoder,
    HybridEmbedding,
)
from semikb.storage.memory import DemoStore


def payload(document_id: str = "T4-TEST-SOP") -> dict[str, object]:
    return {
        "document_id": document_id,
        "revision": "R1",
        "title": "T4 controlled ingestion test",
        "document_type": "sop",
        "content": "# Alarm handling\n\nVerify chamber pressure before recipe recovery.",
        "approval_status": "approved",
        "lifecycle": "published",
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
        "tool_id": "ETCH-03",
    }


class FailOnceEncoder:
    def __init__(self, dimension: int) -> None:
        self._delegate = DeterministicHybridEncoder(dimension)
        self._failed = False

    def encode(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        if not self._failed:
            self._failed = True
            raise RuntimeError("injected embedding failure")
        return self._delegate.encode(texts)


class FailOnceAfterStageStore(DemoStore):
    def __init__(self) -> None:
        super().__init__()
        self._failed = False

    def stage_document(self, document, chunks, images, embeddings) -> None:
        super().stage_document(document, chunks, images, embeddings)
        if not self._failed:
            self._failed = True
            raise RuntimeError("injected staging failure")


def test_retry_replays_persisted_source_after_service_recreation() -> None:
    settings = Settings(_env_file=None, demo_mode=True, embedding_dim=8)
    store = DemoStore()
    first_service = IngestionService(store, settings, encoder=FailOnceEncoder(8))

    failed = first_service.ingest_payload(payload())

    assert failed.status is IngestionStatus.FAILED
    assert store.get_document("T4-TEST-SOP", "R1") is None

    restarted_service = IngestionService(
        store,
        settings,
        encoder=DeterministicHybridEncoder(8),
    )
    completed = restarted_service.retry(failed.job_id)

    assert completed.status is IngestionStatus.PUBLISHED
    assert completed.attempt == 2
    assert store.get_document("T4-TEST-SOP", "R1").lifecycle is DocumentLifecycle.PUBLISHED
    assert {event.attempt for event in completed.events} == {1, 2}


def test_staging_failure_quarantines_records_and_retry_does_not_duplicate() -> None:
    settings = Settings(_env_file=None, demo_mode=True, embedding_dim=8)
    store = FailOnceAfterStageStore()
    service = IngestionService(store, settings, encoder=DeterministicHybridEncoder(8))

    failed = service.ingest_payload(payload("T4-STAGE-FAIL"))

    assert failed.status is IngestionStatus.FAILED
    assert store.get_document("T4-STAGE-FAIL", "R1").lifecycle is DocumentLifecycle.QUARANTINED
    assert not any(
        chunk.lifecycle is DocumentLifecycle.PUBLISHED
        for chunk in store.chunks.values()
        if chunk.document_id == "T4-STAGE-FAIL"
    )

    completed = service.retry(failed.job_id)
    repeated = service.ingest_payload(payload("T4-STAGE-FAIL"))

    assert completed.status is IngestionStatus.PUBLISHED
    assert repeated.job_id == completed.job_id
    assert len(
        [chunk for chunk in store.chunks.values() if chunk.document_id == "T4-STAGE-FAIL"]
    ) == completed.chunks_count


def test_low_confidence_image_caption_fails_quality_gate() -> None:
    settings = Settings(_env_file=None, demo_mode=True, embedding_dim=8)
    store = DemoStore()
    service = IngestionService(store, settings, encoder=DeterministicHybridEncoder(8))
    request = payload("T4-BAD-IMAGE")
    request["images"] = [
        {
            "image_id": "T4-BAD-IMAGE-001",
            "image_type": "wafer_map",
            "caption": "",
            "caption_source": "mineru",
            "caption_confidence": 0.0,
            "source_path": "data/assets/wafer_maps/etch03_chamber_b_edge_ring.png",
        }
    ]

    job = service.ingest_payload(request)

    assert job.status is IngestionStatus.FAILED
    assert store.get_document("T4-BAD-IMAGE", "R1") is None


def test_mineru_archive_returns_markdown_and_referenced_image() -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("result/full.md", "# Inspection\n\n![edge ring](images/wafer.png)")
        archive.writestr("result/images/wafer.png", b"\x89PNG\r\n\x1a\nsynthetic")
        archive.writestr("result/images/unreferenced.png", b"ignored")

    parsed = MinerUPrecisionClient.read_archive(output.getvalue())

    assert parsed.markdown.startswith("# Inspection")
    assert len(parsed.images) == 1
    assert parsed.images[0].filename == "wafer.png"
    assert parsed.images[0].caption == "edge ring"


def test_source_and_parse_objects_are_replayable() -> None:
    settings = Settings(_env_file=None, demo_mode=True, embedding_dim=8)
    store = DemoStore()
    service = IngestionService(store, settings, encoder=DeterministicHybridEncoder(8))

    job = service.ingest_payload(payload("T4-OBJECT-REF"))

    assert job.source_ref is not None
    assert job.parsed_ref is not None
    assert store.load_object(job.source_ref).startswith(b"# Alarm handling")
    assert store.load_object(job.parsed_ref).startswith(b"# Alarm handling")
    assert Path(job.source_ref.object_key).name == "T4-OBJECT-REF-R1.md"
