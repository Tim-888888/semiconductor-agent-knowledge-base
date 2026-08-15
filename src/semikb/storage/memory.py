"""Deterministic in-memory repository used by the runnable synthetic demo."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from semikb.contracts.models import (
    ActiveConversationContext,
    ActorScope,
    AffectSignals,
    ApprovalStatus,
    AuditEvent,
    ChatMessage,
    Chunk,
    DocumentLifecycle,
    DocumentRevision,
    EvaluationDataset,
    EvaluationRun,
    EvaluationStatus,
    ImageAsset,
    IngestionEvent,
    IngestionJob,
    IngestionStatus,
    ObjectRef,
    RetrievalTrace,
    TableAsset,
    ThreadRecord,
)
from semikb.contracts.streaming import (
    AgentMessageRequestRecord,
    AgentMessageRequestStatus,
    AgentStreamErrorCode,
    UnderstandingAudit,
)
from semikb.rag_retrieval.encoders import HybridEmbedding
from semikb.storage.conversations import (
    MessageRequestConflictError,
    MessageRequestInProgressError,
    ThreadBusyError,
)


class DemoStore:
    """A small repository with the same authority boundaries as production storage."""

    def __init__(self) -> None:
        self._lock = RLock()
        self.documents: dict[tuple[str, str], DocumentRevision] = {}
        self.chunks: dict[str, Chunk] = {}
        self.images: dict[str, ImageAsset] = {}
        self.tables: dict[str, TableAsset] = {}
        self.jobs: dict[str, IngestionJob] = {}
        self.job_keys: dict[str, str] = {}
        self.traces: dict[str, RetrievalTrace] = {}
        self.threads: dict[str, ThreadRecord] = {}
        self.message_requests: dict[tuple[str, str, str], AgentMessageRequestRecord] = {}
        self.evaluation_datasets: dict[str, EvaluationDataset] = {}
        self.evaluation_runs: dict[str, EvaluationRun] = {}
        self.index_releases: dict[str, dict[str, str]] = {}
        self.objects: dict[tuple[str, str], bytes] = {}
        self.replay_payloads: dict[str, dict[str, object]] = {}
        self.audit_events: dict[str, AuditEvent] = {}

    def add_document(
        self,
        document: DocumentRevision,
        chunks: list[Chunk],
        images: list[ImageAsset],
        tables: Sequence[TableAsset] = (),
    ) -> None:
        with self._lock:
            self.documents[(document.document_id, document.revision)] = document
            self.chunks.update({chunk.chunk_id: chunk for chunk in chunks})
            self.images.update({image.image_id: image for image in images})
            self.tables.update({table.table_id: table for table in tables})

    def stage_document(
        self,
        document: DocumentRevision,
        chunks: list[Chunk],
        images: list[ImageAsset],
        embeddings: list[HybridEmbedding],
        tables: Sequence[TableAsset] = (),
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Every staged chunk requires one embedding.")
        self.add_document(document, chunks, images, tables)

    def get_document(self, document_id: str, revision: str) -> DocumentRevision | None:
        return self.documents.get((document_id, revision))

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def get_image(self, image_id: str) -> ImageAsset | None:
        return self.images.get(image_id)

    def get_table(self, table_id: str) -> TableAsset | None:
        return self.tables.get(table_id)

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
            job.events.append(
                IngestionEvent(
                    job_id=job.job_id,
                    stage=IngestionStatus.QUEUED,
                    message="Ingestion job accepted.",
                    attempt=job.attempt,
                    progress=0,
                )
            )
            return job

    def save_replay_payload(self, job_id: str, payload: dict[str, object]) -> None:
        self.replay_payloads[job_id] = payload

    def get_replay_payload(self, job_id: str) -> dict[str, object] | None:
        return self.replay_payloads.get(job_id)

    def prepare_retry(self, job_id: str) -> IngestionJob:
        job = self.jobs[job_id]
        if job.status is not IngestionStatus.FAILED:
            return job
        job.attempt += 1
        job.status = IngestionStatus.QUEUED
        job.current_stage = IngestionStatus.QUEUED
        job.progress = 0
        job.error_code = None
        job.safe_error_summary = None
        job.failed_stage = None
        job.finished_at = None
        job.events.append(
            IngestionEvent(
                job_id=job.job_id,
                stage=IngestionStatus.QUEUED,
                message="Ingestion retry accepted.",
                attempt=job.attempt,
                progress=0,
            )
        )
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
            job.events.append(
                IngestionEvent(
                    job_id=job.job_id,
                    stage=stage,
                    message=message,
                    attempt=job.attempt,
                    progress=progress,
                )
            )
            if stage is IngestionStatus.VALIDATING and job.started_at is None:
                job.started_at = datetime.now(UTC)
            if stage is IngestionStatus.FAILED:
                job.error_code = error_code or "INGESTION_FAILED"
                job.safe_error_summary = message
                job.failed_stage = previous_stage
                job.finished_at = datetime.now(UTC)
            elif stage is IngestionStatus.PUBLISHED:
                job.finished_at = datetime.now(UTC)
            return job

    def set_job_artifacts(
        self,
        job_id: str,
        *,
        source_ref: ObjectRef | None = None,
        parsed_ref: ObjectRef | None = None,
    ) -> IngestionJob:
        job = self.jobs[job_id]
        if source_ref is not None:
            job.source_ref = source_ref
        if parsed_ref is not None:
            job.parsed_ref = parsed_ref
        return job

    def set_job_counts(
        self,
        job_id: str,
        *,
        chunks_count: int,
        images_count: int,
        tables_count: int,
    ) -> IngestionJob:
        job = self.jobs[job_id]
        job.chunks_count = chunks_count
        job.images_count = images_count
        job.tables_count = tables_count
        return job

    def set_job_parse_audit(
        self,
        job_id: str,
        *,
        parse_contract_version: str,
        parser_name: str,
        parser_version: str,
        provider_name: str | None,
        provider_version: str | None,
        upstream_project: str | None,
        upstream_commit: str | None,
        chunker_version: str,
        warning_codes: list[str],
        metrics: dict[str, object],
    ) -> IngestionJob:
        job = self.jobs[job_id]
        job.parse_contract_version = parse_contract_version
        job.parser_name = parser_name
        job.parser_version = parser_version
        job.provider_name = provider_name
        job.provider_version = provider_version
        job.upstream_project = upstream_project
        job.upstream_commit = upstream_commit
        job.chunker_version = chunker_version
        job.parse_warning_codes = list(warning_codes)
        job.parse_metrics = dict(metrics)
        return job

    def _store_object(self, object_ref: ObjectRef, content: bytes) -> ObjectRef:
        self.objects[(object_ref.bucket, object_ref.object_key)] = content
        return object_ref

    def store_source(
        self,
        *,
        document_id: str,
        revision: str,
        filename: str,
        content: bytes,
        content_type: str,
        source_hash: str,
    ) -> ObjectRef:
        object_ref = ObjectRef(
            bucket="semikb-raw",
            object_key=f"documents/{document_id}/{revision}/source/{source_hash}/{filename}",
            content_type=content_type,
            sha256=source_hash,
        )
        return self._store_object(object_ref, content)

    def load_object(self, object_ref: ObjectRef) -> bytes:
        return self.objects[(object_ref.bucket, object_ref.object_key)]

    def store_parsed_markdown(
        self,
        *,
        document_id: str,
        revision: str,
        parser_version: str,
        source_hash: str,
        content: bytes,
    ) -> ObjectRef:
        object_ref = ObjectRef(
            bucket="semikb-derived",
            object_key=(
                f"documents/{document_id}/{revision}/parse/{parser_version}/"
                f"{source_hash}/document.md"
            ),
            content_type="text/markdown",
            sha256=hashlib.sha256(content).hexdigest(),
        )
        return self._store_object(object_ref, content)

    def store_image_asset(
        self,
        *,
        document_id: str,
        revision: str,
        image_id: str,
        filename: str,
        content: bytes,
        content_type: str,
        source_hash: str,
    ) -> ObjectRef:
        suffix = Path(filename).suffix.lower() or ".bin"
        object_ref = ObjectRef(
            bucket="semikb-derived",
            object_key=f"documents/{document_id}/{revision}/assets/{image_id}/original{suffix}",
            content_type=content_type,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        return self._store_object(object_ref, content)

    def store_table_asset(
        self,
        *,
        document_id: str,
        revision: str,
        table_id: str,
        content: bytes,
        source_hash: str,
    ) -> ObjectRef:
        object_ref = ObjectRef(
            bucket="semikb-derived",
            object_key=f"documents/{document_id}/{revision}/assets/{table_id}/table.json",
            content_type="application/json",
            sha256=hashlib.sha256(content).hexdigest(),
        )
        return self._store_object(object_ref, content)

    def publish_document(
        self,
        document: DocumentRevision,
        chunks: list[Chunk],
        images: list[ImageAsset],
        embeddings: list[HybridEmbedding],
        tables: Sequence[TableAsset] = (),
    ) -> None:
        with self._lock:
            stored_document = self.documents[(document.document_id, document.revision)]
            stored_document.lifecycle = DocumentLifecycle.PUBLISHED
            stored_document.index_version = document.index_version
            for chunk in self.chunks.values():
                if chunk.document_id == document.document_id and chunk.revision == document.revision:
                    chunk.lifecycle = DocumentLifecycle.PUBLISHED
                    chunk.index_version = document.index_version
            for image in self.images.values():
                if image.document_id == document.document_id and image.revision == document.revision:
                    image.lifecycle = DocumentLifecycle.PUBLISHED
            for table in self.tables.values():
                if table.document_id == document.document_id and table.revision == document.revision:
                    table.lifecycle = DocumentLifecycle.PUBLISHED
            self.index_releases[document.index_version] = {
                "status": "active",
                "alias": "semikb_chunks_active",
            }

    def finalize_inactive_document(
        self,
        document_id: str,
        revision: str,
        lifecycle: DocumentLifecycle,
    ) -> None:
        self.documents[(document_id, revision)].lifecycle = lifecycle
        for chunk in self.chunks.values():
            if chunk.document_id == document_id and chunk.revision == revision:
                chunk.lifecycle = lifecycle
        for image in self.images.values():
            if image.document_id == document_id and image.revision == revision:
                image.lifecycle = lifecycle
        for table in self.tables.values():
            if table.document_id == document_id and table.revision == revision:
                table.lifecycle = lifecycle

    def compensate_document(self, document_id: str, revision: str) -> None:
        if (document_id, revision) in self.documents:
            self.finalize_inactive_document(
                document_id,
                revision,
                DocumentLifecycle.QUARANTINED,
            )

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
        with self._lock:
            current = self.threads.get(thread.thread_id)
            if current is not None and current.active_request_id is not None:
                raise ThreadBusyError(thread.thread_id)
            thread.updated_at = datetime.now(UTC)
            self.threads[thread.thread_id] = thread
            return thread

    def prepare_message_request(
        self,
        record: AgentMessageRequestRecord,
    ) -> tuple[AgentMessageRequestRecord, bool]:
        key = (record.thread_id, record.actor_user_id, record.request_id)
        with self._lock:
            existing = self.message_requests.get(key)
            if existing is None:
                thread = self.threads.get(record.thread_id)
                if thread is None or thread.actor_scope.user_id != record.actor_user_id:
                    raise KeyError(record.thread_id)
                if thread.active_request_id is not None:
                    raise ThreadBusyError(record.thread_id)
                record.user_turn_seq = thread.next_turn_seq
                thread.last_turn_seq = thread.next_turn_seq
                thread.next_turn_seq += 1
                thread.active_request_id = record.request_id
                thread.active_request_started_at = datetime.now(UTC)
                self.message_requests[key] = record
                return record, False
            if existing.content_sha256 != record.content_sha256:
                raise MessageRequestConflictError(record.request_id)
            if existing.status is AgentMessageRequestStatus.COMPLETED:
                return existing, True
            if existing.status in {
                AgentMessageRequestStatus.ACCEPTED,
                AgentMessageRequestStatus.RUNNING,
            }:
                raise MessageRequestInProgressError(record.request_id)
            thread = self.threads.get(record.thread_id)
            if thread is None or thread.actor_scope.user_id != record.actor_user_id:
                raise KeyError(record.thread_id)
            if thread.active_request_id is not None:
                raise ThreadBusyError(record.thread_id)
            thread.active_request_id = record.request_id
            thread.active_request_started_at = datetime.now(UTC)
            existing.status = AgentMessageRequestStatus.ACCEPTED
            existing.attempt += 1
            existing.run_id = record.run_id
            existing.assistant_message_id = None
            existing.trace_id = None
            existing.result_payload = {}
            existing.interaction_mode = None
            existing.route_decision = None
            existing.route_confidence = None
            existing.task_items = []
            existing.task_decisions = []
            existing.task_results = []
            existing.context_message_ids = []
            existing.standalone_query = ""
            existing.retrieval_skipped_reason = None
            existing.slot_operations = []
            existing.inherited_slots = {}
            existing.invalidated_context_refs = []
            existing.cancel_scope = None
            existing.affect = AffectSignals()
            existing.understanding_audit = UnderstandingAudit()
            existing.error_code = None
            existing.updated_at = datetime.now(UTC)
            existing.finished_at = None
            return existing, False

    def get_message_request(
        self,
        thread_id: str,
        actor_user_id: str,
        request_id: str,
    ) -> AgentMessageRequestRecord | None:
        return self.message_requests.get((thread_id, actor_user_id, request_id))

    def list_message_requests(
        self,
        thread_id: str,
        actor_user_id: str,
    ) -> list[AgentMessageRequestRecord]:
        return sorted(
            (
                record
                for (stored_thread_id, stored_user_id, _), record in self.message_requests.items()
                if stored_thread_id == thread_id and stored_user_id == actor_user_id
            ),
            key=lambda record: record.created_at,
        )

    def append_message_once(self, thread_id: str, message: ChatMessage) -> ThreadRecord:
        with self._lock:
            thread = self.threads.get(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            if message.request_id and thread.active_request_id != message.request_id:
                raise ThreadBusyError(thread_id)
            duplicate = message.request_id and any(
                item.request_id == message.request_id and item.role == message.role
                for item in thread.messages
            )
            if not duplicate:
                thread.messages.append(message)
                thread.updated_at = datetime.now(UTC)
            return thread

    def finalize_stream_response(
        self,
        thread_id: str,
        message: ChatMessage,
        *,
        status: str,
        summary: str,
        summary_upto_message_id: str | None,
        active_context: ActiveConversationContext,
        context_version: int,
        pending_fields: list[str],
        clarification_round: int,
    ) -> ThreadRecord:
        with self._lock:
            thread = self.threads.get(thread_id)
            if thread is None:
                raise KeyError(thread_id)
            if not any(
                item.request_id == message.request_id and item.role == "assistant"
                for item in thread.messages
            ):
                if thread.active_request_id != message.request_id:
                    raise ThreadBusyError(thread_id)
                message.turn_seq = thread.next_turn_seq
                thread.last_turn_seq = thread.next_turn_seq
                thread.next_turn_seq += 1
                thread.messages.append(message)
            thread.status = status
            thread.summary = summary
            thread.summary_upto_message_id = summary_upto_message_id
            thread.active_context = active_context
            thread.context_version = context_version
            thread.pending_fields = pending_fields
            thread.clarification_round = clarification_round
            thread.updated_at = datetime.now(UTC)
            return thread

    def mark_message_request_running(
        self,
        record: AgentMessageRequestRecord,
    ) -> AgentMessageRequestRecord:
        with self._lock:
            current = self.get_message_request(
                record.thread_id,
                record.actor_user_id,
                record.request_id,
            )
            if current is None or current.status is not AgentMessageRequestStatus.ACCEPTED:
                raise MessageRequestInProgressError(record.request_id)
            current.status = AgentMessageRequestStatus.RUNNING
            current.updated_at = datetime.now(UTC)
            return current

    def mark_message_request_terminal(
        self,
        record: AgentMessageRequestRecord,
        status: AgentMessageRequestStatus,
        *,
        result_payload: dict[str, object] | None = None,
        assistant_message_id: str | None = None,
        trace_id: str | None = None,
        error_code: AgentStreamErrorCode | None = None,
    ) -> AgentMessageRequestRecord:
        if status not in {
            AgentMessageRequestStatus.COMPLETED,
            AgentMessageRequestStatus.FAILED,
            AgentMessageRequestStatus.CANCELLED,
        }:
            raise ValueError("terminal request status required")
        with self._lock:
            current = self.get_message_request(
                record.thread_id,
                record.actor_user_id,
                record.request_id,
            )
            if current is None:
                raise KeyError(record.request_id)
            if current.status not in {
                AgentMessageRequestStatus.ACCEPTED,
                AgentMessageRequestStatus.RUNNING,
                status,
            }:
                raise RuntimeError("message request terminal transition was not acknowledged")
            current.status = status
            current.result_payload = result_payload or {}
            current.assistant_message_id = assistant_message_id
            current.assistant_turn_seq = record.assistant_turn_seq
            current.trace_id = trace_id
            current.interaction_mode = record.interaction_mode
            current.route_decision = record.route_decision
            current.route_confidence = record.route_confidence
            current.task_items = list(record.task_items)
            current.task_decisions = list(record.task_decisions)
            current.task_results = list(record.task_results)
            current.context_message_ids = list(record.context_message_ids)
            current.standalone_query = record.standalone_query
            current.retrieval_skipped_reason = record.retrieval_skipped_reason
            current.slot_operations = list(record.slot_operations)
            current.inherited_slots = dict(record.inherited_slots)
            current.invalidated_context_refs = list(record.invalidated_context_refs)
            current.cancel_scope = record.cancel_scope
            current.affect = record.affect.model_copy(deep=True)
            current.understanding_audit = record.understanding_audit.model_copy(deep=True)
            current.error_code = error_code
            current.updated_at = datetime.now(UTC)
            current.finished_at = current.updated_at
            thread = self.threads.get(record.thread_id)
            if thread is not None and thread.active_request_id == record.request_id:
                thread.active_request_id = None
                thread.active_request_started_at = None
            return current

    def append_audit(self, event: AuditEvent) -> AuditEvent:
        self.audit_events.setdefault(event.event_id, event)
        return event

    def save_evaluation_run(self, run: EvaluationRun) -> EvaluationRun:
        self.evaluation_runs[run.evaluation_run_id] = run
        return run

    def save_evaluation_dataset(self, dataset: EvaluationDataset) -> EvaluationDataset:
        existing = self.evaluation_datasets.get(dataset.dataset_version)
        if existing and existing.dataset_hash != dataset.dataset_hash:
            raise ValueError(
                f"Dataset version {dataset.dataset_version!r} already has a different hash."
            )
        self.evaluation_datasets.setdefault(dataset.dataset_version, dataset)
        return self.evaluation_datasets[dataset.dataset_version]

    def get_evaluation_dataset(self, dataset_version: str) -> EvaluationDataset | None:
        return self.evaluation_datasets.get(dataset_version)

    def list_evaluation_datasets(self) -> list[EvaluationDataset]:
        return sorted(
            self.evaluation_datasets.values(),
            key=lambda dataset: dataset.created_at,
            reverse=True,
        )

    def claim_evaluation_run(
        self,
        evaluation_run_id: str,
        execution_id: str | None = None,
    ) -> EvaluationRun | None:
        run = self.evaluation_runs.get(evaluation_run_id)
        if run is None:
            return None
        claimable = run.status is EvaluationStatus.QUEUED or (
            bool(execution_id)
            and run.status is EvaluationStatus.RUNNING
            and run.worker_task_id == execution_id
        )
        if not claimable:
            return None
        run.status = EvaluationStatus.RUNNING
        run.started_at = datetime.now(UTC)
        run.worker_task_id = execution_id
        return self.save_evaluation_run(run)

    def prepare_evaluation_retry(self, evaluation_run_id: str) -> EvaluationRun:
        run = self.evaluation_runs.get(evaluation_run_id)
        if run is None:
            raise KeyError(evaluation_run_id)
        if run.status is not EvaluationStatus.FAILED:
            return run
        run.status = EvaluationStatus.QUEUED
        run.started_at = None
        run.finished_at = None
        run.safe_error_summary = None
        run.worker_task_id = None
        run.attempt += 1
        return self.save_evaluation_run(run)

    def get_evaluation_run(self, evaluation_run_id: str) -> EvaluationRun | None:
        return self.evaluation_runs.get(evaluation_run_id)

    def list_evaluation_runs(self) -> list[EvaluationRun]:
        return sorted(self.evaluation_runs.values(), key=lambda run: run.created_at, reverse=True)
