# Semiconductor Agent Knowledge Base

An evidence-driven semiconductor knowledge assistant for controlled SOP, Recipe,
Case, inspection image, and simulated manufacturing-data investigation workflows.

## Current scope

Phase 1 provides a runnable demo path for document ingestion, version-aware hybrid
retrieval, synthetic wafer-image assets, continuous threads, retrieval traces, offline
evaluation, and an operations-focused web UI. It never controls equipment or writes
manufacturing data.

| Capability | Phase 1 runnable Demo | External-service activation |
| --- | --- | --- |
| Ingestion | Markdown/fixture ingestion, SHA-256 idempotency, quality gate, semantic chunks | Celery Worker, MinerU adapter, BGE-M3 and MinIO/Mongo/Milvus production repositories are implemented and T4 live acceptance has passed |
| Retrieval | Version/ACL-aware Dense + Sparse + RRF + deterministic rerank and cutoff | Local BGE-M3/BGE reranker and Milvus hybrid indexes |
| Conversation | LangGraph branch control, two clarification rounds, continuous `thread_id` | MongoDB LangGraph checkpointer and OpenAI-compatible response client |
| Operations | Trace, golden-set evaluation, task-centre UI | Redis/Celery worker, real service topology and production observability |

## Local development

1. Use the `dl` Conda environment with Python 3.12.
2. Copy `.env.example` to `.env` and fill service values when external services are ready.
3. Keep `DEMO_MODE=true` to run the synthetic-data workflow without external services.
4. Install backend dependencies with `pip install -e .[dev,rag]`.
5. Start the API with `uvicorn semikb.api.main:app --reload --port 8000`.
6. Install and start the web application from `web/` with `npm install` and `npm run dev`.

For the complete synthetic demonstration, run `scripts/run_demo.ps1` from PowerShell.
For external storage, run `python -m semikb.storage.preflight` before provisioning and
`python -m semikb.storage.verifier` to validate the full resource contract.
The repeatable real-storage T4 acceptance entry point is
`python scripts/verify_t4_ingestion.py`.

See `docs/` for storage, API, security, testing, and deployment decisions.
