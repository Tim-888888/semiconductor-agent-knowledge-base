"""Task entry points with idempotent service-level behavior."""

from semikb.bootstrap import get_container
from semikb.workers.celery_app import celery_app


@celery_app.task(name="semikb.ingestion.retry", autoretry_for=(ConnectionError,), retry_backoff=True, max_retries=3)
def retry_ingestion_job(job_id: str) -> dict[str, object]:
    job = get_container().ingestion.retry(job_id)
    return job.model_dump(mode="json")


@celery_app.task(name="semikb.evaluation.run", autoretry_for=(ConnectionError,), retry_backoff=True, max_retries=2)
def run_evaluation(dataset_version: str = "demo-v1", baseline_run_id: str | None = None) -> dict[str, object]:
    run = get_container().evaluation.run(dataset_version, baseline_run_id)
    return run.model_dump(mode="json")
