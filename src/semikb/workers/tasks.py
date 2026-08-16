"""Task entry points with idempotent service-level behavior."""

from semikb.bootstrap import get_container
from semikb.contracts.corpus import CorpusStandardizationStatus
from semikb.contracts.corpus_publication import CorpusPublicationStatus
from semikb.contracts.models import (
    DocumentLifecycleOperationStatus,
    IngestionStatus,
)
from semikb.rag_ingestion.corpus_standardization import TRANSIENT_CORPUS_FAILURE_CODES
from semikb.workers.celery_app import celery_app

_TRANSIENT_INGESTION_ERRORS = {
    "CONNECTIONERROR",
    "INGEST_PARSER_TIMEOUT",
    "INGEST_PARSER_UNAVAILABLE",
    "MILVUSEXCEPTION",
    "MINERUERROR",
    "S3ERROR",
    "SERVERSELECTIONTIMEOUTERROR",
    "TIMEOUTEXCEPTION",
}

_TRANSIENT_EVALUATION_ERRORS = {
    "APITIMEOUTERROR",
    "CONNECTIONERROR",
    "MILVUSEXCEPTION",
    "RERANKERERROR",
    "SERVERSELECTIONTIMEOUTERROR",
    "S3ERROR",
    "TIMEOUTERROR",
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


@celery_app.task(
    bind=True,
    name="semikb.corpus.standardize",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    soft_time_limit=1500,
    time_limit=1800,
)
def process_corpus_standardization(self, job_id: str) -> dict[str, object]:
    service = get_container().corpus_standardization
    existing = service.get_job(job_id)
    if existing is None:
        raise KeyError(job_id)
    if (
        existing.status is CorpusStandardizationStatus.FAILED
        and (existing.error_code or "").upper() in TRANSIENT_CORPUS_FAILURE_CODES
    ):
        existing = service.prepare_retry(job_id)
    job = service.process(existing.job_id)
    if (
        job.status is CorpusStandardizationStatus.FAILED
        and (job.error_code or "").upper() in TRANSIENT_CORPUS_FAILURE_CODES
        and self.request.retries < self.max_retries
    ):
        service.prepare_retry(job_id)
        raise self.retry(
            exc=ConnectionError("Transient corpus standardization failure."),
            countdown=2 ** (self.request.retries + 1),
        )
    return job.model_dump(mode="json")


@celery_app.task(
    bind=True,
    name="semikb.corpus.publish",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    soft_time_limit=1800,
    time_limit=2100,
)
def process_corpus_publication(self, batch_id: str) -> dict[str, object]:
    service = get_container().corpus_publication
    existing = service.get(batch_id)
    if existing is None:
        raise KeyError(batch_id)
    if existing.status is CorpusPublicationStatus.FAILED and existing.error_code in {
        "CONNECTIONERROR",
        "MILVUSEXCEPTION",
        "S3ERROR",
        "SERVERSELECTIONTIMEOUTERROR",
        "TIMEOUTERROR",
    }:
        existing = service.prepare_retry(batch_id)
    batch = service.process(existing.batch_id, execution_id=self.request.id)
    if (
        batch.status is CorpusPublicationStatus.FAILED
        and batch.error_code
        in {
            "CONNECTIONERROR",
            "MILVUSEXCEPTION",
            "S3ERROR",
            "SERVERSELECTIONTIMEOUTERROR",
            "TIMEOUTERROR",
        }
        and self.request.retries < self.max_retries
    ):
        service.prepare_retry(batch_id)
        raise self.retry(
            exc=ConnectionError("Transient corpus publication failure."),
            countdown=2 ** (self.request.retries + 1),
        )
    return batch.model_dump(mode="json")


@celery_app.task(
    bind=True,
    name="semikb.knowledge_document.lifecycle",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=3,
    soft_time_limit=1200,
    time_limit=1500,
)
def process_document_lifecycle_operation(
    self,
    operation_id: str,
) -> dict[str, object]:
    service = get_container().knowledge_documents
    operation = service.process(operation_id, execution_id=self.request.id)
    if (
        operation.status is DocumentLifecycleOperationStatus.COMPENSATION_REQUIRED
        and self.request.retries < self.max_retries
    ):
        service.prepare_retry(operation_id)
        raise self.retry(
            exc=ConnectionError("Lifecycle projection requires compensation."),
            countdown=2 ** (self.request.retries + 1),
        )
    return operation.model_dump(mode="json")


@celery_app.task(
    bind=True,
    name="semikb.evaluation.run",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=2,
    soft_time_limit=1500,
    time_limit=1800,
)
def run_evaluation(self, evaluation_run_id: str) -> dict[str, object]:
    service = get_container().evaluation
    try:
        run = service.execute(evaluation_run_id, execution_id=self.request.id)
    except Exception as exc:
        transient = type(exc).__name__.upper() in _TRANSIENT_EVALUATION_ERRORS
        if transient and self.request.retries < self.max_retries:
            service.prepare_retry(evaluation_run_id)
            raise self.retry(exc=exc, countdown=2 ** (self.request.retries + 1))
        raise
    return run.model_dump(mode="json")
