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
| Retrieval | Version/ACL-aware Dense + Sparse + RRF + deterministic rerank and cutoff | Live BGE-M3/Milvus + conditional Luna HyDE + qwen3-rerank pipeline passed T5 synthetic acceptance |
| Conversation | LangGraph `interrupt/resume`, two clarification rounds, evidence ledger and explicit memory | MongoDB Checkpointer/Store plus Luna primary/Qwen fallback answer generation passed restart acceptance |
| Operations | Trace, golden-set evaluation, task-centre UI | Real API/Worker-backed Trace, ingestion, evaluation and responsive browser workflows passed T8 acceptance; production observability remains T9 scope |

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
The credential-safe primary LLM smoke test is
`python scripts/verify_llm_gateway.py`.
The live T5 baseline comparison is
`python scripts/verify_t5_retrieval.py`.
The live T6 interrupt/resume acceptance is
`python scripts/verify_t6_agent.py`.
The live T7 evaluation acceptance is
`python scripts/verify_t7_evaluation.py --timeout 900 --replace-acceptance-runs`.
The T8 browser click and screenshot record is
`docs/T8浏览器验收.md`.

See `docs/` for storage, API, security, testing, and deployment decisions.
