"""HTTP API for the Phase 1 synthetic demo and real-service adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from semikb.api.auth import create_demo_token, get_actor_scope
from semikb.bootstrap import ApplicationContainer, get_container
from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    CreateEvaluationRunRequest,
    CreateMemoryRequest,
    CreateThreadRequest,
    IngestDocumentRequest,
    IngestionStatus,
    IngestUploadMetadata,
    SearchRequest,
    SendMessageRequest,
)
from semikb.rag_retrieval.milvus_schema import schema_contract
from semikb.storage.external import health_payload


def get_app_container() -> ApplicationContainer:
    return get_container()


def get_app_settings(container: Annotated[ApplicationContainer, Depends(get_app_container)]) -> Settings:
    return container.settings


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


@app.post("/api/v1/auth/demo-token")
def issue_demo_token(
    scope: ActorScope,
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> dict[str, str]:
    return {"access_token": create_demo_token(scope, settings), "token_type": "bearer"}


@app.post("/api/v1/demo/seed")
def seed_demo(container: Annotated[ApplicationContainer, Depends(get_app_container)]) -> dict[str, int]:
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
    if container.settings.demo_mode:
        job = container.ingestion.ingest_payload(payload, created_by=actor_scope.user_id)
    else:
        job = container.ingestion.submit_payload(payload, created_by=actor_scope.user_id)
        _enqueue_ingestion(container, job.job_id)
    return job.model_dump(mode="json")


@app.post("/api/v1/ingestion-jobs/upload", status_code=status.HTTP_201_CREATED)
async def upload_ingestion_document(
    file: Annotated[UploadFile, File(...)],
    metadata: Annotated[str, Form(...)],
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    """Upload a source document; binary formats are normalized through MinerU."""

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
    if len(source_bytes) > 200 * 1024 * 1024:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds the 200 MiB limit.")
    filename = file.filename or "upload.bin"
    ingestion_method = (
        container.ingestion.ingest_file
        if container.settings.demo_mode
        else container.ingestion.submit_file
    )
    job = await run_in_threadpool(
        ingestion_method,
        filename,
        source_bytes,
        upload_metadata.model_dump(mode="json"),
        actor_scope.user_id,
    )
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


@app.post("/api/v1/evaluation-runs", status_code=status.HTTP_201_CREATED)
def create_evaluation_run(
    request: CreateEvaluationRunRequest,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    actor_scope: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    if "admin" not in actor_scope.roles and "knowledge_admin" not in actor_scope.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Knowledge administrator role required.")
    run = container.evaluation.run(request.dataset_version, request.baseline_run_id)
    return run.model_dump(mode="json")


@app.get("/api/v1/evaluation-runs")
def list_evaluation_runs(
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    _: Annotated[ActorScope, Depends(get_actor_scope)],
) -> list[dict[str, object]]:
    return [run.model_dump(mode="json") for run in container.store.list_evaluation_runs()]


@app.get("/api/v1/evaluation-runs/{evaluation_run_id}")
def get_evaluation_run(
    evaluation_run_id: str,
    container: Annotated[ApplicationContainer, Depends(get_app_container)],
    _: Annotated[ActorScope, Depends(get_actor_scope)],
) -> dict[str, object]:
    run = container.store.get_evaluation_run(evaluation_run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evaluation run not found.")
    return run.model_dump(mode="json")
