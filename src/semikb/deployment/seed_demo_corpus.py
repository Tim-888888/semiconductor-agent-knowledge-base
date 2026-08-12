"""Idempotently ingest the public synthetic corpus into freshly provisioned storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semikb.config import get_settings
from semikb.contracts.models import IngestionStatus
from semikb.rag_ingestion.service import IngestionService
from semikb.storage.production_ingestion import ProductionIngestionStore


def load_corpus(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Synthetic corpus must contain a non-empty documents list.")
    return documents


def seed(path: Path) -> dict[str, object]:
    settings = get_settings()
    if settings.demo_mode:
        raise RuntimeError("Server seeding requires DEMO_MODE=false.")
    service = IngestionService(ProductionIngestionStore(settings), settings)
    jobs = [service.ingest_payload(document, created_by="deployment_seed") for document in load_corpus(path)]
    failed = [job.job_id for job in jobs if job.status is IngestionStatus.FAILED]
    if failed:
        raise RuntimeError(f"Synthetic corpus seeding failed for {len(failed)} job(s).")
    return {
        "status": "seeded",
        "documents": len(jobs),
        "published_jobs": sum(job.status is IngestionStatus.PUBLISHED for job in jobs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--fixture", type=Path, default=Path("data/fixtures/demo_corpus.json"))
    args = parser.parse_args()
    documents = load_corpus(args.fixture)
    if not args.apply:
        print(json.dumps({"status": "dry_run", "documents": len(documents)}, ensure_ascii=False))
        return
    print(json.dumps(seed(args.fixture), ensure_ascii=False))


if __name__ == "__main__":
    main()
