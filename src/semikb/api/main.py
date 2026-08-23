"""HTTP API for the Phase 1 synthetic demo and real-service adapters."""

from __future__ import annotations

import asyncio
import hmac
import json
from pathlib import Path
from typing import Annotated

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse

from semikb.api.auth import create_demo_token, get_actor_scope, resolve_demo_token_scope
from semikb.bootstrap import ApplicationContainer, get_container
from semikb.config import Settings
from semikb.contracts.corpus import (
    CorpusSidecar,
    CorpusStandardizationError,
    CorpusStandardizationMetadata,
    CorpusStandardizationStatus,
    CorpusUploadedFile,
)
from semikb.contracts.corpus_publication import (
    CorpusPublicationReview,
    CorpusPublicationStatus,
)
from semikb.contracts.evaluation_governance import (
    CreateEvaluationReleaseFreezeRequest,
    RegisterEvaluationDatasetRequest,
)
from semikb.contracts.models import (
    ActorScope,
    CreateEvaluationRunRequest,
    CreateMemoryRequest,
    CreateThreadRequest,
    DocumentLifecycle,
    DocumentLifecycleOperationRecord,
    DocumentLifecycleOperationStatus,
    DocumentRevisionSelector,
    EvaluationDataset,
    EvaluationStatus,
    IngestDocumentRequest,
    IngestionStatus,
    IngestUploadMetadata,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentRevisionSummary,
    RestoreDocumentRevisionRequest,
    SearchRequest,
    SendMessageRequest,
    WithdrawDocumentRevisionRequest,
)
from semikb.contracts.streaming import StreamMessageRequest, encode_sse_event
from semikb.rag_ingestion.corpus_publication import CorpusPublicationError
from semikb.rag_ingestion.document_lifecycle import (
    LifecycleStateConflictError,
    LifecycleValidationError,
)
from semikb.rag_ingestion.service import IngestionIdempotencyConflictError
from semikb.rag_retrieval.milvus_schema import schema_contract
from semikb.storage.conversations import (
    MessageRequestConflictError,
    MessageRequestInProgressError,
    ThreadBusyError,
)
from semikb.storage.corpus_publication import CorpusPublicationConflictError
from semikb.storage.corpus_standardization import CorpusStandardizationConflictError
from semikb.storage.external import health_payload
from semikb.storage.knowledge_documents import (
    LifecycleOperationRequestConflictError,
    RevisionLifecycleConflictError,
)
from semikb_ingest import IngestError


def get_app_container() -> ApplicationContainer:
    return get_container()


def get_app_settings(container: Annotated[ApplicationContainer, Depends(get_app_container)]) -> Settings:
    return container.settings


def _require_knowledge_admin(actor_scope: ActorScope) -> None:
    if "admin" not in actor_scope.roles and "knowledge_admin" not in actor_scope.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Knowledge administrator role required.",
        )


def _evaluation_dataset_payload(dataset: EvaluationDataset) -> dict[str, object]:
    payload = dataset.model_dump(mode="json")
    if dataset.purpose.value == "holdout" and dataset.opened_at is None:
        payload.pop("cases", None)
        payload["cases_redacted"] = True
    return payload


app = FastAPI(title="Semiconductor Agent Knowledge Base", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health(container: Annotated[ApplicationContainer, Depends(get_app_container)]) -> dict[str, object]:
    return {
        "status": "ok",
        "demo_mode": container.settings.demo_mode,
        "services": health_payload(container.settings),
        "milvus_schema": schema_contract(
            container.settings.embedding_dim,
            container.settings.milvus_index_version,
        ),
    }


@app.get("/api/v1/live")
def live() -> dict[str, str]:
    """Container liveness probe that does not initialize external clients."""

    return {"status": "ok"}


@app.post("/api/v1/auth/demo-token")
def issue_demo_token(
    settings: Annotated[Settings, Depends(get_app_settings)],
    scope: Annotated[ActorScope | None, Body()] = None,
    x_demo_access_key: Annotated[str | None, Header(alias="X-Demo-Access-Key")] = None,
) -> dict[str, str]:
    if settings.demo_access_key:
        if not x_demo_access_key or not hmac.compare_digest(
            x_demo_access_key,
            settings.demo_access_key,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid demo access key.",
            )
    elif settings.app_env.lower() == "production":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Demo access key is not configured.",
        )
    token_scope = resolve_demo_token_scope(scope, settings)
    return {"access_token": create_demo_token(token_scope, settings), "token_type": "bearer"}


@app.post("/api/v1/demo/seed")
def seed_demo(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, int]:
    if not container.settings.demo_mode:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    _require_knowledge_admin(actor_scope)
    container.seed_demo_data()
    return {
        "documents": len(container.store.documents),
        "chunks": len(container.store.chunks),
        "images": len(container.store.images),
    }


@app.post("/api/v1/threads", status_code=status.HTTP_201_CREATED)
def create_thread(
    request: CreateThreadRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    thread = container.conversations.create_thread(request.title, actor_scope)
    return thread.model_dump(mode="json")


@app.get("/api/v1/threads")
def list_threads(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    return [
        thread.model_dump(mode="json")
        for thread in container.conversations.list_threads(actor_scope)
    ]


@app.get("/api/v1/threads/{thread_id}")
def get_thread(
    thread_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    thread = container.conversations.get_thread(thread_id, actor_scope)
    if thread is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")
    return thread.model_dump(mode="json")


@app.post("/api/v1/threads/{thread_id}/messages")
async def send_message(
    thread_id: str,
    request: SendMessageRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    try:
        return await container.conversations.send_message(thread_id, request.content, actor_scope)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
    except ThreadBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another request is already running in this thread.",
        ) from exc


@app.post("/api/v1/threads/{thread_id}/messages/stream")
async def stream_message(
    thread_id: str,
    stream_request: StreamMessageRequest,
    http_request: Request,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> StreamingResponse:
    """Stream safe Agent progress and verified answer deltas over SSE."""

    try:
        prepared = await container.conversations.prepare_stream_message(
            thread_id,
            stream_request.content,
            stream_request.request_id,
            actor_scope,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.") from exc
    except MessageRequestConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="request_id was already used with different content.",
        ) from exc
    except ThreadBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another request is already running in this thread.",
        ) from exc
    except MessageRequestInProgressError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="request_id is already being processed.",
        ) from exc

    async def event_stream():
        stream = container.conversations.stream_message(prepared)

        async def watch_disconnect() -> None:
            while True:
                if await http_request.is_disconnected():
                    await container.conversations.cancel_stream_message(
                        thread_id,
                        stream_request.request_id,
                        actor_scope,
                    )
                    return
                await asyncio.sleep(0.2)

        disconnect_task = asyncio.create_task(watch_disconnect())

        async def cleanup_stream() -> None:
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
            try:
                await container.conversations.cancel_stream_message(
                    thread_id,
                    stream_request.request_id,
                    actor_scope,
                )
            except KeyError:
                pass
            await stream.aclose()

        try:
            async for event in stream:
                if await http_request.is_disconnected():
                    break
                yield encode_sse_event(event)
        finally:
            cleanup_task = asyncio.create_task(cleanup_stream())
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                await cleanup_task
                raise

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/v1/threads/{thread_id}/message-requests/{request_id}")
async def get_message_request(
    thread_id: str,
    request_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    if container.conversations.get_thread(thread_id, actor_scope) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found.")
    record = await run_in_threadpool(
        container.conversation_store.get_message_request,
        thread_id,
        actor_scope.user_id,
        request_id,
    )
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found.")
    return record.model_dump(
        mode="json",
        exclude={"content_sha256", "result_payload"},
    )


@app.post("/api/v1/threads/{thread_id}/message-requests/{request_id}/cancel")
async def cancel_message_request(
    thread_id: str,
    request_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    try:
        record = await container.conversations.cancel_stream_message(
            thread_id,
            request_id,
            actor_scope,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Request not found.") from exc
    return record.model_dump(
        mode="json",
        exclude={"content_sha256", "result_payload"},
    )


@app.post("/api/v1/memories", status_code=status.HTTP_201_CREATED)
def create_memory(
    request: CreateMemoryRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    try:
        memory = container.conversations.memory.create(request, actor_scope)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    return memory.model_dump(mode="json")


@app.get("/api/v1/memories")
def list_memories(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    return [
        memory.model_dump(mode="json")
        for memory in container.conversations.memory.list(actor_scope)
    ]


@app.delete("/api/v1/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(
    memory_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> Response:
    try:
        container.conversations.memory.delete(memory_id, actor_scope)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/v1/retrieval/search")
def search(
    request: SearchRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    evidence, trace = container.retrieval.search(
        request.query,
        actor_scope,
        top_k=request.top_k,
        thread_id=request.thread_id,
        constraints=request.constraints,
    )
    return {
        "evidence": [chunk.model_dump(mode="json") for chunk in evidence],
        "trace": trace.model_dump(mode="json"),
    }


@app.post("/api/v1/ingestion-jobs", status_code=status.HTTP_201_CREATED)
def create_ingestion_job(
    request: IngestDocumentRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    if "admin" not in actor_scope.roles and "knowledge_admin" not in actor_scope.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Knowledge administrator role required.")
    payload = request.model_dump(mode="json")
    try:
        if container.settings.demo_mode:
            job = container.ingestion.ingest_payload(payload, created_by=actor_scope.user_id)
        else:
            job = container.ingestion.submit_payload(payload, created_by=actor_scope.user_id)
            _enqueue_ingestion(container, job.job_id)
    except IngestError as exc:
        raise _ingest_http_exception(exc) from exc
    except IngestionIdempotencyConflictError as exc:
        raise _ingestion_idempotency_conflict() from exc
    return job.model_dump(mode="json")


@app.post("/api/v1/ingestion-jobs/upload", status_code=status.HTTP_201_CREATED)
async def upload_ingestion_document(
    file: Annotated[UploadFile, File(...)],
    metadata: Annotated[str, Form(...)],
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    """Upload a source document through the exact-format parser registry."""

    if "admin" not in actor_scope.roles and "knowledge_admin" not in actor_scope.roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Knowledge administrator role required.",
        )
    try:
        upload_metadata = IngestUploadMetadata.model_validate(json.loads(metadata))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid metadata JSON.") from exc
    source_bytes = await file.read()
    if not source_bytes:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file is empty.")
    if len(source_bytes) > container.settings.max_upload_mib * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {container.settings.max_upload_mib} MiB limit.",
        )
    filename = file.filename or "upload.bin"
    ingestion_method = (
        container.ingestion.ingest_file
        if container.settings.demo_mode
        else container.ingestion.submit_file
    )
    try:
        job = await run_in_threadpool(
            ingestion_method,
            filename,
            source_bytes,
            upload_metadata.model_dump(mode="json"),
            actor_scope.user_id,
            content_type=file.content_type,
        )
    except IngestError as exc:
        raise _ingest_http_exception(exc) from exc
    except IngestionIdempotencyConflictError as exc:
        raise _ingestion_idempotency_conflict() from exc
    if not container.settings.demo_mode:
        _enqueue_ingestion(container, job.job_id)
    return job.model_dump(mode="json")


@app.get("/api/v1/ingestion-jobs")
def list_ingestion_jobs(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    _: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    return [job.model_dump(mode="json") for job in container.ingestion.list_jobs()]


@app.get("/api/v1/ingestion-jobs/{job_id}")
def get_ingestion_job(
    job_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    _: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    job = container.ingestion.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.")
    return job.model_dump(mode="json")


@app.post("/api/v1/ingestion-jobs/{job_id}/retry")
def retry_ingestion_job(
    job_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    if "admin" not in actor_scope.roles and "knowledge_admin" not in actor_scope.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Knowledge administrator role required.")
    try:
        if container.settings.demo_mode:
            job = container.ingestion.retry(job_id)
        else:
            job = container.ingestion.prepare_retry(job_id)
            _enqueue_ingestion(container, job.job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found.") from exc
    return job.model_dump(mode="json")


@app.post("/api/v1/corpus-standardization-jobs/upload", status_code=status.HTTP_201_CREATED)
async def upload_corpus_standardization_job(
    files: Annotated[list[UploadFile], File(...)],
    metadata: Annotated[str, Form(...)],
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
    sidecar: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    """Create an immutable multi-file snapshot and prepare it for human review."""

    _require_knowledge_admin(actor_scope)
    try:
        parsed_metadata = CorpusStandardizationMetadata.model_validate(json.loads(metadata))
        parsed_sidecar = (
            CorpusSidecar.model_validate(json.loads(sidecar)) if sidecar else CorpusSidecar()
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid corpus metadata or sidecar JSON.",
        ) from exc
    uploads: list[CorpusUploadedFile] = []
    total_bytes = 0
    for file in files:
        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Corpus files cannot be empty.",
            )
        total_bytes += len(content)
        if total_bytes > container.settings.max_upload_mib * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Corpus upload exceeds the {container.settings.max_upload_mib} MiB limit.",
            )
        uploads.append(
            CorpusUploadedFile(
                relative_path=file.filename or "upload.bin",
                content_type=file.content_type or "application/octet-stream",
                content=content,
            )
        )
    try:
        job = await run_in_threadpool(
            container.corpus_standardization.submit,
            uploads,
            parsed_metadata,
            parsed_sidecar,
            actor_scope.user_id,
        )
        if container.settings.demo_mode:
            job = await run_in_threadpool(container.corpus_standardization.process, job.job_id)
        else:
            _enqueue_corpus_standardization(container, job.job_id)
    except CorpusStandardizationConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except CorpusStandardizationError as exc:
        code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if exc.code == "CORPUS_UPLOAD_LIMIT_EXCEEDED"
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(
            status_code=code,
            detail={"code": exc.code, "message": exc.safe_message},
        ) from exc
    return job.model_dump(mode="json")


@app.get("/api/v1/corpus-standardization-jobs")
def list_corpus_standardization_jobs(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    _require_knowledge_admin(actor_scope)
    return [
        job.model_dump(mode="json")
        for job in container.corpus_standardization.list_jobs()
    ]


@app.get("/api/v1/corpus-standardization-jobs/{job_id}")
def get_corpus_standardization_job(
    job_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    job = container.corpus_standardization.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus standardization job not found.",
        )
    return job.model_dump(mode="json")


@app.post("/api/v1/corpus-standardization-jobs/{job_id}/retry")
def retry_corpus_standardization_job(
    job_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    try:
        job = container.corpus_standardization.prepare_retry(job_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus standardization job not found.",
        ) from exc
    if container.settings.demo_mode:
        job = container.corpus_standardization.process(job.job_id)
    else:
        _enqueue_corpus_standardization(container, job.job_id)
    return job.model_dump(mode="json")


@app.post("/api/v1/corpus-publication-batches", status_code=status.HTTP_202_ACCEPTED)
def create_corpus_publication_batch(
    review: CorpusPublicationReview,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    try:
        batch = container.corpus_publication.submit(
            review,
            created_by=actor_scope.user_id,
        )
        if container.settings.demo_mode:
            batch = container.corpus_publication.process(batch.batch_id)
        else:
            _enqueue_corpus_publication(container, batch.batch_id)
    except CorpusPublicationConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except CorpusPublicationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": exc.code, "message": exc.safe_message},
        ) from exc
    return batch.model_dump(mode="json")


@app.get("/api/v1/corpus-publication-batches")
def list_corpus_publication_batches(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    _require_knowledge_admin(actor_scope)
    return [
        batch.model_dump(mode="json") for batch in container.corpus_publication.list()
    ]


@app.get("/api/v1/corpus-publication-batches/{batch_id}")
def get_corpus_publication_batch(
    batch_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    batch = container.corpus_publication.get(batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus publication batch not found.",
        )
    return batch.model_dump(mode="json")


@app.post("/api/v1/corpus-publication-batches/{batch_id}/retry")
def retry_corpus_publication_batch(
    batch_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    try:
        batch = container.corpus_publication.prepare_retry(batch_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus publication batch not found.",
        ) from exc
    if container.settings.demo_mode:
        batch = container.corpus_publication.process(batch.batch_id)
    else:
        _enqueue_corpus_publication(container, batch.batch_id)
    return batch.model_dump(mode="json")


@app.get(
    "/api/v1/knowledge-documents",
    response_model=KnowledgeDocumentListResponse,
)
def list_knowledge_documents(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
    query: Annotated[str | None, Query(max_length=200)] = None,
    lifecycle: Annotated[DocumentLifecycle | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> KnowledgeDocumentListResponse:
    _require_knowledge_admin(actor_scope)
    return container.knowledge_documents.list_documents(
        actor_scope,
        query=query,
        lifecycle=lifecycle,
        limit=limit,
        offset=offset,
    )


@app.get(
    "/api/v1/knowledge-documents/{document_id}/revisions",
    response_model=list[KnowledgeDocumentRevisionSummary],
)
def list_knowledge_document_revisions(
    document_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[KnowledgeDocumentRevisionSummary]:
    _require_knowledge_admin(actor_scope)
    return container.knowledge_documents.list_revisions(document_id, actor_scope)


@app.post(
    "/api/v1/knowledge-documents/{document_id}/revisions/{revision}/withdraw",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentLifecycleOperationRecord,
)
def withdraw_knowledge_document_revision(
    document_id: str,
    revision: str,
    request: WithdrawDocumentRevisionRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> DocumentLifecycleOperationRecord:
    _require_knowledge_admin(actor_scope)
    try:
        operation = container.knowledge_documents.request_withdrawal(
            DocumentRevisionSelector(document_id=document_id, revision=revision),
            request,
            actor_scope,
        )
    except Exception as exc:
        raise _lifecycle_http_exception(exc) from exc
    operation = _dispatch_lifecycle_operation(container, operation.operation_id)
    return operation


@app.post(
    "/api/v1/knowledge-documents/{document_id}/revisions/{revision}/restore",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentLifecycleOperationRecord,
)
def restore_knowledge_document_revision(
    document_id: str,
    revision: str,
    request: RestoreDocumentRevisionRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> DocumentLifecycleOperationRecord:
    _require_knowledge_admin(actor_scope)
    try:
        operation = container.knowledge_documents.request_restore(
            DocumentRevisionSelector(document_id=document_id, revision=revision),
            request,
            actor_scope,
        )
    except Exception as exc:
        raise _lifecycle_http_exception(exc) from exc
    operation = _dispatch_lifecycle_operation(container, operation.operation_id)
    return operation


@app.get(
    "/api/v1/knowledge-document-operations/{operation_id}",
    response_model=DocumentLifecycleOperationRecord,
)
def get_knowledge_document_operation(
    operation_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> DocumentLifecycleOperationRecord:
    _require_knowledge_admin(actor_scope)
    operation = container.knowledge_documents.get_operation(operation_id)
    if operation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge document operation not found.",
        )
    if "admin" not in actor_scope.roles and operation.actor_user_id != actor_scope.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Knowledge document operation access denied.",
        )
    return operation


@app.post(
    "/api/v1/knowledge-document-operations/{operation_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DocumentLifecycleOperationRecord,
)
def retry_knowledge_document_operation(
    operation_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> DocumentLifecycleOperationRecord:
    _require_knowledge_admin(actor_scope)
    existing = container.knowledge_documents.get_operation(operation_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge document operation not found.",
        )
    if "admin" not in actor_scope.roles and existing.actor_user_id != actor_scope.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Knowledge document operation access denied.",
        )
    operation = container.knowledge_documents.prepare_retry(operation_id)
    if operation.status not in {
        DocumentLifecycleOperationStatus.WITHDRAWN,
        DocumentLifecycleOperationStatus.RESTORED,
    }:
        operation = _dispatch_lifecycle_operation(container, operation.operation_id)
    return operation


def _dispatch_lifecycle_operation(
    container: ApplicationContainer,
    operation_id: str,
) -> DocumentLifecycleOperationRecord:
    if container.settings.demo_mode:
        return container.knowledge_documents.process(operation_id)
    from semikb.workers.tasks import process_document_lifecycle_operation

    try:
        process_document_lifecycle_operation.delay(operation_id)
    except Exception:
        return container.knowledge_documents.mark_dispatch_failed(operation_id)
    operation = container.knowledge_documents.get_operation(operation_id)
    if operation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge document operation not found.",
        )
    return operation


def _lifecycle_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge document revision not found.",
        )
    if isinstance(exc, PermissionError):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Knowledge document revision access denied.",
        )
    if isinstance(exc, LifecycleValidationError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    if isinstance(
        exc,
        (
            LifecycleStateConflictError,
            LifecycleOperationRequestConflictError,
            RevisionLifecycleConflictError,
        ),
    ):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    raise exc


def _enqueue_ingestion(container: ApplicationContainer, job_id: str) -> None:
    from semikb.workers.tasks import process_ingestion_job

    job = container.ingestion.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ingestion job not found.",
        )
    if job.status is not IngestionStatus.QUEUED:
        return
    try:
        process_ingestion_job.delay(job_id)
    except Exception as exc:
        container.ingestion.mark_queue_submission_failed(job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion task queue is unavailable. Retry the failed job later.",
        ) from exc


def _enqueue_corpus_standardization(
    container: ApplicationContainer,
    job_id: str,
) -> None:
    from semikb.workers.tasks import process_corpus_standardization

    job = container.corpus_standardization.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus standardization job not found.",
        )
    if job.status is not CorpusStandardizationStatus.QUEUED:
        return
    try:
        process_corpus_standardization.delay(job_id)
    except Exception as exc:
        container.corpus_standardization.mark_queue_submission_failed(job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Corpus standardization queue is unavailable. Retry the failed job later.",
        ) from exc


def _enqueue_corpus_publication(
    container: ApplicationContainer,
    batch_id: str,
) -> None:
    from semikb.workers.tasks import process_corpus_publication

    batch = container.corpus_publication.get(batch_id)
    if batch is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Corpus publication batch not found.",
        )
    if batch.status is not CorpusPublicationStatus.QUEUED:
        return
    try:
        process_corpus_publication.delay(batch_id)
    except Exception as exc:
        container.corpus_publication.mark_queue_submission_failed(batch_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Corpus publication queue is unavailable. Retry the failed batch later.",
        ) from exc


def _ingest_http_exception(exc: IngestError) -> HTTPException:
    return HTTPException(
        status_code=exc.descriptor.http_status,
        detail={"code": exc.code.value, "message": exc.safe_message},
    )


def _ingestion_idempotency_conflict() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": "INGESTION_IDEMPOTENCY_CONFLICT",
            "message": (
                "The same document revision and file were already submitted "
                "with different ingestion metadata."
            ),
        },
    )


@app.get("/api/v1/assets/{image_id}/access")
def get_asset_access(
    image_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, str]:
    try:
        return container.retrieval.asset_access(image_id, actor_scope)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image asset not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Image access denied.") from exc


@app.get("/api/v1/assets/{image_id}/preview")
def preview_asset(
    image_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> Response:
    try:
        access = container.retrieval.asset_access(image_id, actor_scope)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image asset not found.") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Image access denied.") from exc
    if not container.settings.demo_mode:
        return RedirectResponse(access["url"], status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    asset = container.store.get_image(image_id)
    if asset and asset.demo_source_path:
        root = Path(__file__).resolve().parents[3]
        allowed_root = (root / "data" / "assets").resolve()
        asset_path = (root / asset.demo_source_path).resolve()
        if not asset_path.is_relative_to(allowed_root) or not asset_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image asset file not found.")
        return FileResponse(asset_path, media_type=asset.object_ref.content_type)

    svg = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 480 480'>
<rect width='480' height='480' fill='#14212b'/><circle cx='240' cy='240' r='174' fill='#dce6eb'/>
<circle cx='240' cy='240' r='142' fill='none' stroke='#c44945' stroke-width='21'/>
<circle cx='240' cy='240' r='104' fill='#b7c7d1'/><path d='M130 236c44-28 175-29 220 0' fill='none' stroke='#738a98' stroke-width='4'/>
<text x='240' y='442' text-anchor='middle' font-family='Arial' font-size='18' fill='#eef5f7'>Synthetic wafer edge-ring inspection image</text>
</svg>"""
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/v1/retrieval-traces")
def list_retrieval_traces(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    return [trace.model_dump(mode="json") for trace in container.retrieval.list_traces(actor_scope)]


@app.get("/api/v1/retrieval-traces/{trace_id}")
def get_retrieval_trace(
    trace_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    trace = container.retrieval.get_trace(trace_id, actor_scope)
    if trace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Retrieval trace not found.")
    return trace.model_dump(mode="json")


@app.post("/api/v1/evaluation-runs", status_code=status.HTTP_202_ACCEPTED)
def create_evaluation_run(
    request: CreateEvaluationRunRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    try:
        run = container.evaluation.create_run(
            request.dataset_version,
            request.baseline_run_id,
            retrieval_profile=request.retrieval_profile,
            requested_by=actor_scope.user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    if container.settings.demo_mode:
        run = container.evaluation.execute(run.evaluation_run_id)
    else:
        _enqueue_evaluation(container, run.evaluation_run_id)
    return run.model_dump(mode="json")


@app.get("/api/v1/evaluation-datasets")
def list_evaluation_datasets(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    _require_knowledge_admin(actor_scope)
    return [
        _evaluation_dataset_payload(dataset)
        for dataset in container.evaluation.list_datasets()
    ]


@app.post("/api/v1/evaluation-datasets", status_code=status.HTTP_201_CREATED)
def register_evaluation_dataset(
    request: RegisterEvaluationDatasetRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    try:
        dataset = container.evaluation.register_dataset(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    return _evaluation_dataset_payload(dataset)


@app.post("/api/v1/evaluation-release-freezes", status_code=status.HTTP_201_CREATED)
def create_evaluation_release_freeze(
    request: CreateEvaluationReleaseFreezeRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    try:
        freeze = container.evaluation.create_release_freeze(
            request,
            created_by=actor_scope.user_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return freeze.model_dump(mode="json")


@app.get("/api/v1/evaluation-release-freezes")
def list_evaluation_release_freezes(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    _require_knowledge_admin(actor_scope)
    return [
        freeze.model_dump(mode="json")
        for freeze in container.evaluation.list_release_freezes()
    ]


@app.get("/api/v1/evaluation-runs")
def list_evaluation_runs(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    _require_knowledge_admin(actor_scope)
    return [run.model_dump(mode="json") for run in container.evaluation.list_runs()]


@app.get("/api/v1/evaluation-runs/{evaluation_run_id}")
def get_evaluation_run(
    evaluation_run_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    run = container.evaluation.get_run(evaluation_run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found.")
    return run.model_dump(mode="json")


@app.get("/api/v1/evaluation-runs/{evaluation_run_id}/cases/{case_id}/trace")
def get_evaluation_case_trace(
    evaluation_run_id: str,
    case_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    run = container.evaluation.get_run(evaluation_run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found.")
    case_result = next(
        (result for result in run.case_results if result.get("case_id") == case_id),
        None,
    )
    if case_result is None or not case_result.get("trace_id"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation case trace not found.",
        )
    trace = container.retrieval.get_trace(str(case_result["trace_id"]), None)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation case trace not found.",
        )
    return trace.model_dump(mode="json")


@app.post("/api/v1/evaluation-runs/{evaluation_run_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_evaluation_run(
    evaluation_run_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    _require_knowledge_admin(actor_scope)
    existing = container.evaluation.get_run(evaluation_run_id)
    if existing is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found.")
    if existing.status is not EvaluationStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only a failed evaluation run can be retried.",
        )
    run = container.evaluation.prepare_retry(evaluation_run_id)
    if container.settings.demo_mode:
        run = container.evaluation.execute(evaluation_run_id)
    else:
        _enqueue_evaluation(container, evaluation_run_id)
    return run.model_dump(mode="json")


def _enqueue_evaluation(container: ApplicationContainer, evaluation_run_id: str) -> None:
    from semikb.workers.tasks import run_evaluation

    run = container.evaluation.get_run(evaluation_run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation run not found.",
        )
    if run.status is not EvaluationStatus.QUEUED:
        return
    try:
        run_evaluation.delay(evaluation_run_id)
    except Exception as exc:
        container.evaluation.mark_queue_submission_failed(evaluation_run_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evaluation task queue is unavailable. Retry the failed run later.",
        ) from exc
