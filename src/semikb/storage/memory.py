"""Deterministic in-memory repository used by the runnable synthetic demo."""

from __future__ import annotations

from datetime import UTC, datetime
from threading import RLock

from semikb.contracts.models import (
    ActorScope,
    ApprovalStatus,
    Chunk,
    DocumentLifecycle,
    DocumentRevision,
    EvaluationRun,
    ImageAsset,
    IngestionEvent,
    IngestionJob,
    IngestionStatus,
    RetrievalTrace,
    ThreadRecord,
)


class DemoStore:
    """A small repository with the same authority boundaries as production storage."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.documents: dict[tuple[str, str], DocumentRevision] = {}
        self.chunks: dict[str, Chunk] = {}
        self.images: dict[str, ImageAsset] = {}
        self.jobs: dict[str, IngestionJob] = {}
        self.job_keys: dict[str, str] = {}
        self.traces: dict[str, RetrievalTrace] = {}
        self.threads: dict[str, ThreadRecord] = {}
        self.evaluation_runs: dict[str, EvaluationRun] = {}
        self.index_releases: dict[str, dict[str, str]] = {}

    def add_document(self, document: DocumentRevision, chunks: list[Chunk], images: list[ImageAsset]) -> None:
        with self._lock:
            self.documents[(document.document_id, document.revision)] = document
            self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})
            self.images.update({image.image_id: image for image in images})

    def get_document(self, document_id: str, revision: str) -> DocumentRevision | None:
        return self.documents.get((document_id, revision))

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_image(self, image_id: str) -> ImageAsset | None:
        return self.images.get(image_id)

    def list_published_chunks(self, actor_scope: ActorScope, now: datetime | None = None) -> list[Chunk]:
        current = now or datetime.now(UTC)
        return [
            chunk
            for chunk in self.chunks.values()
            if self._is_accessible(chunk, actor_scope, current)
        ]

    @staticmethod
    def _is_accessible(chunk: Chunk, actor_scope: ActorScope, current: datetime) -> bool:
        if "admin" in actor_scope.roles:
            return True
        if chunk.lifecycle is not DocumentLifecycle.PUBLISHED:
            return False
        if chunk.approval_status is not ApprovalStatus.APPROVED:
            return False
        if chunk.effective_at > current or (chunk.expires_at and chunk.expires_at <= current):
            return False
        if chunk.access_scope_key not in actor_scope.access_scope_keys:
            return False
        if chunk.fab and actor_scope.fabs and chunk.fab not in actor_scope.fabs:
            return False
        if chunk.product and actor_scope.products and chunk.product not in actor_scope.products:
            return False
        return not (chunk.tool_id and actor_scope.tool_ids and chunk.tool_id not in actor_scope.tool_ids)

    def create_or_get_job(self, job: IngestionJob) -> IngestionJob:
        with self._lock:
            existing_id = self.job_keys.get(job.idempotency_key)
            if existing_id:
                return self.jobs[existing_id]
            self.jobs[job.job_id] = job
            self.job_keys[job.idempotency_key] = job.job_id
            return job

    def get_job(self, job_id: str) -> IngestionJob | None:
        return self.jobs.get(job_id)

    def list_jobs(self) -> list[IngestionJob]:
        return sorted(self.jobs.values(), key=lambda job: job.created_at, reverse=True)

    def update_job(
        self,
        job_id: str,
        stage: IngestionStatus,
        message: str,
        progress: int,
        *,
        error_code: str | None = None,
    ) -> IngestionJob:
        with self._lock:
            job = self.jobs[job_id]
            previous_stage = job.current_stage
            job.status = stage
            job.current_stage = stage
            job.progress = progress
            job.events.append(IngestionEvent(stage=stage, message=message))
            if stage is IngestionStatus.FAILED:
                job.error_code = error_code or "INGESTION_FAILED"
                job.safe_error_summary = message
                job.failed_stage = previous_stage
                job.finished_at = datetime.now(UTC)
            elif stage is IngestionStatus.PUBLISHED:
                job.finished_at = datetime.now(UTC)
            return job

    def publish_document(self, document_id: str, revision: str, index_version: str) -> None:
        with self._lock:
            document = self.documents[(document_id, revision)]
            document.lifecycle = DocumentLifecycle.PUBLISHED
            document.index_version = index_version
            for chunk in self.chunks.values():
                if chunk.document_id == document_id and chunk.revision == revision:
                    chunk.lifecycle = DocumentLifecycle.PUBLISHED
                    chunk.index_version = index_version
            for image in self.images.values():
                if image.document_id == document_id and image.revision == revision:
                    image.lifecycle = DocumentLifecycle.PUBLISHED
            self.index_releases[index_version] = {"status": "active", "alias": "semikb_chunks_active"}

    def save_trace(self, trace: RetrievalTrace) -> RetrievalTrace:
        self.traces[trace.trace_id] = trace
        return trace

    def get_trace(self, trace_id: str, actor_scope: ActorScope | None = None) -> RetrievalTrace | None:
        trace = self.traces.get(trace_id)
        if trace is None or actor_scope is None:
            return trace
        if "admin" in actor_scope.roles or trace.actor_user_id == actor_scope.user_id:
            return trace
        return None

    def list_traces(self, actor_scope: ActorScope | None = None) -> list[RetrievalTrace]:
        traces = self.traces.values()
        if actor_scope is not None and "admin" not in actor_scope.roles:
            traces = (trace for trace in traces if trace.actor_user_id == actor_scope.user_id)
        return sorted(traces, key=lambda trace: trace.created_at, reverse=True)

    def create_thread(self, thread: ThreadRecord) -> ThreadRecord:
        self.threads[thread.thread_id] = thread
        return thread

    def get_thread(self, thread_id: str) -> ThreadRecord | None:
        return self.threads.get(thread_id)

    def list_threads(self, user_id: str) -> list[ThreadRecord]:
        return sorted(
            (thread for thread in self.threads.values() if thread.actor_scope.user_id == user_id),
            key=lambda thread: thread.updated_at,
            reverse=True,
        )

    def save_thread(self, thread: ThreadRecord) -> ThreadRecord:
        thread.updated_at = datetime.now(UTC)
        self.threads[thread.thread_id] = thread
        return thread

    def save_evaluation_run(self, run: EvaluationRun) -> EvaluationRun:
        self.evaluation_runs[run.evaluation_run_id] = run
        return run

    def get_evaluation_run(self, evaluation_run_id: str) -> EvaluationRun | None:
        return self.evaluation_runs.get(evaluation_run_id)

    def list_evaluation_runs(self) -> list[EvaluationRun]:
        return sorted(self.evaluation_runs.values(), key=lambda run: run.created_at, reverse=True)
