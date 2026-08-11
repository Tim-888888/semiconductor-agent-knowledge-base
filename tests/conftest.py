from __future__ import annotations

from pathlib import Path

import pytest

from semikb.agent_runtime.service import ConversationService
from semikb.config import Settings
from semikb.evaluation.service import EvaluationService
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.service import RetrievalService
from semikb.storage.memory import DemoStore


@pytest.fixture
def seeded_services() -> tuple[DemoStore, IngestionService, RetrievalService, ConversationService, EvaluationService]:
    store = DemoStore()
    ingestion = IngestionService(store)
    root = Path(__file__).resolve().parents[1]
    ingestion.seed_demo_corpus(root / "data" / "fixtures" / "demo_corpus.json")
    retrieval = RetrievalService(store)
    settings = Settings(demo_mode=True)
    conversation = ConversationService(store, retrieval, settings)
    evaluation = EvaluationService(store, retrieval, root / "data" / "golden_sets")
    return store, ingestion, retrieval, conversation, evaluation
