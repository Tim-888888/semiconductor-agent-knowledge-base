"""Task entry points with idempotent service-level behavior."""

from semikb.bootstrap import get_container
from semikb.contracts.models import IngestionStatus
from semikb.workers.celery_app import celery_app

_TRANSIENT_INGESTION_ERRORS = {
    "CONNECTIONERROR",
    "MILVUSEXCEPTION",
    "MINERUERROR",
    "S3ERROR",
    "SERVERSELECTIONTIMEOUTERROR",
    "TIMEOUTEXCEPTION",
}


@celery_app.task(
    name="semikb.ingestion.process",
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
)
def process_ingestion_job(job_id: str) -> dict[str, object]:
    service = get_container().ingestion
    existing = service.get_job(job_id)
    if existing is None:
        raise KeyError(job_id)
    if existing.status is IngestionStatus.FAILED and existing.error_code in _TRANSIENT_INGESTION_ERRORS:
        service.prepare_retry(job_id)
    job = service.process(job_id)
    if job.status is IngestionStatus.FAILED and job.error_code in _TRANSIENT_INGESTION_ERRORS:
        raise ConnectionError("Transient ingestion failure; Celery will retry.")
    return job.model_dump(mode="json")


@celery_app.task(name="semikb.ingestion.retry")
def retry_ingestion_job(job_id: str) -> dict[str, object]:
    job = get_container().ingestion.prepare_retry(job_id)
    process_ingestion_job.delay(job.job_id)
    return job.model_dump(mode="json")


@celery_app.task(name="semikb.evaluation.run", autoretry_for=(ConnectionError,), retry_backoff=True, max_retries=2)
def run_evaluation(dataset_version: str = "demo-v2", baseline_run_id: str | None = None) -> dict[str, object]:
    run = get_container().evaluation.run(dataset_version, baseline_run_id)
    return run.model_dump(mode="json")
