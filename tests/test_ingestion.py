from __future__ import annotations

from semikb.contracts.models import ActorScope, IngestionStatus


def test_seeded_ingestion_is_idempotent_and_expired_revision_is_not_retrievable(seeded_services) -> None:
    store, ingestion, _, _, _ = seeded_services
    original_jobs = len(store.jobs)
    repeated_jobs = ingestion.seed_demo_corpus(__import__("pathlib").Path(__file__).resolve().parents[1] / "data" / "fixtures" / "demo_corpus.json")

    assert len(store.jobs) == original_jobs
    assert all(job.status is IngestionStatus.PUBLISHED for job in repeated_jobs)
    ids = {chunk.chunk_id for chunk in store.list_published_chunks(ActorScope())}
    assert not any(chunk_id.startswith("SOP-ETCH-03-R1") for chunk_id in ids)
    assert any(chunk_id.startswith("SOP-ETCH-03-R2") for chunk_id in ids)


def test_uploaded_document_without_effective_at_is_processed(seeded_services) -> None:
    _, ingestion, _, _, _ = seeded_services
    job = ingestion.ingest_payload(
        {
            "document_id": "TRAINING-ETCH-01",
            "revision": "R1",
            "title": "Training note",
            "document_type": "training_note",
            "content": "# Training\n\nCheck pressure alarm before changing a recipe.",
            "lifecycle": "published",
        }
    )

    assert job.status is IngestionStatus.PUBLISHED
    assert job.chunks_count == 1
