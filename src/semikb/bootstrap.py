"""Application composition root shared by FastAPI and Celery tasks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from semikb.agent_runtime.service import ConversationService
from semikb.config import Settings, get_settings
from semikb.evaluation.service import EvaluationService
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.encoders import BgeM3Encoder
from semikb.rag_retrieval.production_service import ProductionRetrievalService
from semikb.rag_retrieval.service import RetrievalService
from semikb.storage.memory import DemoStore
from semikb.storage.production_ingestion import ProductionIngestionStore


class ApplicationContainer:
    """Owns application services; real repositories can replace DemoStore behind this seam."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.store = DemoStore()
        self.ingestion_store = (
            self.store if settings.demo_mode else ProductionIngestionStore(settings)
        )
        shared_encoder = None if settings.demo_mode else BgeM3Encoder(settings)
        self.ingestion = IngestionService(self.ingestion_store, settings, encoder=shared_encoder)
        self.retrieval = (
            RetrievalService(self.store)
            if settings.demo_mode
            else ProductionRetrievalService(settings, encoder=shared_encoder)
        )
        root = Path(__file__).resolve().parents[2]
        self.evaluation = EvaluationService(self.store, self.retrieval, root / "data" / "golden_sets")
        self.conversations = ConversationService(self.store, self.retrieval, settings)
        self._seeded = False

    def seed_demo_data(self) -> None:
        if not self.settings.demo_mode:
            return
        if self._seeded:
            return
        root = Path(__file__).resolve().parents[2]
        self.ingestion.seed_demo_corpus(root / "data" / "fixtures" / "demo_corpus.json")
        self._seeded = True


@lru_cache
def get_container() -> ApplicationContainer:
    container = ApplicationContainer(get_settings())
    if container.settings.demo_mode:
        container.seed_demo_data()
    return container
