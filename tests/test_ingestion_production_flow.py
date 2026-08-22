from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence
from pathlib import Path

import httpx

from semikb.config import Settings
from semikb.contracts.models import (
    Chunk,
    DocumentLifecycle,
    DocumentRevision,
    IngestionStatus,
    ObjectRef,
    TableAsset,
)
from semikb.demo_factory import load_demo_source_manifest
from semikb.rag_ingestion.mineru import MinerUPrecisionClient
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.encoders import (
    DeterministicHybridEncoder,
    HybridEmbedding,
)
from semikb.storage.memory import DemoStore
from semikb.storage.production_ingestion import ProductionIngestionStore
from semikb_provider_resilience import ProviderAttemptAudit


def payload(document_id: str = "T4-TEST-SOP") -> dict[str, object]:
    return {
        "document_id": document_id,
        "revision": "R1",
        "title": "T4 controlled ingestion test",
        "document_type": "sop",
        "content": "# Alarm handling\n\nVerify chamber pressure before recipe recovery.",
        "approval_status": "approved",
        "lifecycle": "published",
        "source_kind": "synthetic",
        "source_license": "CC0-1.0",
        "source_id": "semikb.demo.synthetic",
        "source_manifest_version": "1.0.0",
        "dataset_version": "demo-v2",
        "source_license_status": "verified",
        "redistribution_policy": "allowed",
        "access_scope_key": "demo_engineering",
        "fab": "FAB-01",
        "product": "P-ALPHA",
        "tool_id": "ETCH-03",
        "retrieval_policy": "protected",
    }


def register_demo_manifest(store: DemoStore) -> None:
    store.register_source_manifest(
        load_demo_source_manifest(
            Path("data/source_manifests/semikb-demo-corpus-v1.json")
        )
    )


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
    register_demo_manifest(store)
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
    register_demo_manifest(store)
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
    register_demo_manifest(store)
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


def test_mineru_signed_upload_does_not_override_content_type(monkeypatch) -> None:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("result/full.md", "# Parsed document")

    class FakeClient:
        put_headers: dict[str, str] | None = None

        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def post(self, url, **_kwargs):
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://upload.example/signed"],
                    },
                },
            )

        def put(self, url, **kwargs):
            self.put_headers = kwargs.get("headers")
            return httpx.Response(200, request=httpx.Request("PUT", url))

        def get(self, url, **_kwargs):
            if "extract-results" in url:
                return httpx.Response(
                    200,
                    request=httpx.Request("GET", url),
                    json={
                        "code": 0,
                        "data": {
                            "extract_result": [
                                {
                                    "state": "done",
                                    "full_zip_url": "https://download.example/result.zip",
                                }
                            ]
                        },
                    },
                )
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                content=output.getvalue(),
            )

    fake_client = FakeClient()
    monkeypatch.setattr(
        "semikb.rag_ingestion.mineru.httpx.Client",
        lambda **_kwargs: fake_client,
    )
    settings = Settings(
        _env_file=None,
        mineru_api_base_url="https://mineru.example",
        mineru_api_key="test-key",
    )

    parsed = MinerUPrecisionClient(settings).parse_file(
        "document.pdf",
        b"%PDF-1.4",
        "document:R1",
    )

    assert parsed.markdown == "# Parsed document"
    assert fake_client.put_headers is None


def test_source_and_parse_objects_are_replayable() -> None:
    settings = Settings(_env_file=None, demo_mode=True, embedding_dim=8)
    store = DemoStore()
    register_demo_manifest(store)
    service = IngestionService(store, settings, encoder=DeterministicHybridEncoder(8))

    job = service.ingest_payload(payload("T4-OBJECT-REF"))

    assert job.source_ref is not None
    assert job.parsed_ref is not None
    assert store.load_object(job.source_ref).startswith(b"# Alarm handling")
    assert store.load_object(job.parsed_ref).startswith(b"# Alarm handling")
    assert Path(job.source_ref.object_key).name == "T4-OBJECT-REF-R1.md"


def test_embedding_event_describes_the_active_encoder() -> None:
    settings = Settings(_env_file=None, demo_mode=True, embedding_dim=8)
    store = DemoStore()
    register_demo_manifest(store)
    service = IngestionService(store, settings, encoder=DeterministicHybridEncoder(8))

    job = service.ingest_payload(payload("T4-EMBEDDING-EVENT"))

    embedding_event = next(
        event for event in job.events if event.stage is IngestionStatus.EMBEDDING
    )
    assert embedding_event.message == (
        "Generating Dense and Sparse representations with "
        "deterministic-demo / lexical-hash-demo-v1."
    )
    assert "BGE" not in embedding_event.message


def test_production_publish_records_native_sparse_release_metadata() -> None:
    class Mongo:
        release_kwargs: dict[str, object] | None = None

        def publish_document(self, document) -> None:
            pass

        def record_release(self, document, chunks_count, **kwargs) -> None:
            self.release_kwargs = kwargs

        def supersede_previous(self, document) -> list[str]:
            return []

    class Vectors:
        def upsert_chunks(self, chunks, embeddings, *, lifecycle) -> None:
            assert lifecycle is DocumentLifecycle.PUBLISHED

        def activate_alias(self, index_version: str) -> None:
            assert index_version == "v4"

        def delete_chunks(self, index_version: str, chunk_ids) -> None:
            pass

    settings = Settings(
        _env_file=None,
        demo_mode=False,
        embedding_dim=1024,
        embedding_output_type="dense&sparse",
        sparse_encoder_version="qwen3.7-text-embedding-sparse-v1",
    )
    source_ref = ObjectRef(
        bucket="semikb-raw",
        object_key="test.md",
        content_type="text/markdown",
        sha256="0" * 64,
    )
    document = DocumentRevision(
        document_id="T4-NATIVE-SPARSE",
        revision="R1",
        title="Native Sparse release metadata",
        document_type="sop",
        source_hash="0" * 64,
        source_ref=source_ref,
        embedding_version="qwen3.7-text-embedding+qwen3.7-text-embedding-sparse-v1",
        index_version="v4",
    )
    store = object.__new__(ProductionIngestionStore)
    store._settings = settings
    store.mongo = Mongo()
    store.vectors = Vectors()

    store.publish_document(document, [], [], [])

    assert store.mongo.release_kwargs == {
        "embedding_dim": 1024,
        "embedding_output_type": "dense&sparse",
        "sparse_encoder_version": "qwen3.7-text-embedding-sparse-v1",
        "normalization": "dense_l2_sparse_provider_raw",
    }


def test_production_stage_routes_tables_to_mongo_and_only_chunks_to_milvus() -> None:
    class Mongo:
        staged_tables: list[TableAsset] | None = None

        def stage_document(self, document, chunks, images, tables) -> None:
            self.staged_tables = list(tables)

        def compensate_document(self, document_id, revision) -> list[str]:
            raise AssertionError("successful staging must not compensate")

    class Vectors:
        received_chunks: list[Chunk] | None = None

        def upsert_chunks(self, chunks, embeddings, *, lifecycle) -> None:
            self.received_chunks = list(chunks)
            assert len(chunks) == len(embeddings) == 1
            assert lifecycle is DocumentLifecycle.STAGED

    source_ref = ObjectRef(
        bucket="semikb-raw",
        object_key="source.xlsx",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        sha256="0" * 64,
    )
    table_ref = ObjectRef(
        bucket="semikb-derived",
        object_key="documents/T9444/R1/assets/TABLE-001/table.json",
        content_type="application/json",
        sha256="1" * 64,
    )
    document = DocumentRevision(
        document_id="T9444-PRODUCTION",
        revision="R1",
        title="Production table routing",
        document_type="sop",
        source_hash="0" * 64,
        source_ref=source_ref,
    )
    chunk = Chunk(
        chunk_id="T9444-PRODUCTION-R1-001",
        document_id=document.document_id,
        revision=document.revision,
        chunk_text="| Signal | Limit |\n| --- | --- |\n| Pressure | 12 |",
        page_or_section="Sheet FDC A1:B2",
        table_ids=["T9444-PRODUCTION-R1-TABLE-001"],
    )
    table = TableAsset(
        table_id="T9444-PRODUCTION-R1-TABLE-001",
        document_id=document.document_id,
        revision=document.revision,
        object_ref=table_ref,
        markdown=chunk.chunk_text,
        html="<table><tr><th>Signal</th><th>Limit</th></tr></table>",
        headers=["Signal", "Limit"],
        row_count=1,
        column_count=2,
    )
    embedding = DeterministicHybridEncoder(8).encode([chunk.chunk_text])
    store = object.__new__(ProductionIngestionStore)
    store.mongo = Mongo()
    store.vectors = Vectors()

    store.stage_document(document, [chunk], [], embedding, tables=[table])

    assert store.mongo.staged_tables == [table]
    assert store.vectors.received_chunks == [chunk]


def test_production_store_forwards_embedding_provider_attempt_audit() -> None:
    attempt = ProviderAttemptAudit(
        provider="qianwen-embedding",
        operation="dense_sparse_embedding",
        attempt=1,
        max_attempts=2,
        outcome="succeeded",
        latency_ms=12.5,
    )

    class Mongo:
        received: tuple[str, list[ProviderAttemptAudit], str] | None = None

        def record_provider_attempts(self, job_id, attempts, *, message):
            self.received = (job_id, list(attempts), message)
            return "stored"

    store = object.__new__(ProductionIngestionStore)
    store.mongo = Mongo()

    result = store.record_provider_attempts(
        "ing_test",
        [attempt],
        message="credential-safe audit",
    )

    assert result == "stored"
    assert store.mongo.received == ("ing_test", [attempt], "credential-safe audit")
