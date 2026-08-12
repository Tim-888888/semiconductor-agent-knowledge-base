"""Application composition root shared by FastAPI and Celery tasks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore

from semikb.agent_runtime.service import ConversationService
from semikb.config import Settings, get_settings
from semikb.evaluation.service import EvaluationService
from semikb.rag_ingestion.service import IngestionService
from semikb.rag_retrieval.encoders import create_hybrid_encoder
from semikb.rag_retrieval.production_service import ProductionRetrievalService
from semikb.rag_retrieval.service import RetrievalService
from semikb.storage.conversations import MongoConversationRepository
from semikb.storage.evaluations import MongoEvaluationRepository
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
        shared_encoder = None if settings.demo_mode else create_hybrid_encoder(settings)
        self.ingestion = IngestionService(self.ingestion_store, settings, encoder=shared_encoder)
        self.retrieval = (
            RetrievalService(self.store)
            if settings.demo_mode
            else ProductionRetrievalService(settings, encoder=shared_encoder)
        )
        root = Path(__file__).resolve().parents[2]
        self.evaluation_store = (
            self.store if settings.demo_mode else MongoEvaluationRepository(settings)
        )
        self.evaluation = EvaluationService(
            self.evaluation_store,
            self.retrieval,
            root / "data" / "golden_sets",
            settings,
        )
        if settings.demo_mode:
            self.conversation_store = self.store
            self.conversations = ConversationService(self.store, self.retrieval, settings)
        else:
            self.conversation_store = MongoConversationRepository(settings)
            checkpointer = MongoDBSaver(
                self.conversation_store.client,
                db_name=settings.mongodb_database,
                checkpoint_collection_name="checkpoints",
                writes_collection_name="checkpoint_writes",
            )
            long_term_store = MongoDBStore(
                collection=self.conversation_store.database["long_term_memories"]
            )
            self.conversations = ConversationService(
                self.conversation_store,
                self.retrieval,
                settings,
                checkpointer=checkpointer,
                long_term_store=long_term_store,
            )
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
