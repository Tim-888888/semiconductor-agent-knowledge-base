from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    Chunk,
    ChunkType,
    DocumentLifecycle,
    RetrievalCandidate,
    RetrievalConstraints,
    RetrievalTrace,
)
from semikb.rag_retrieval.encoders import HybridEmbedding
from semikb.rag_retrieval.production_repository import (
    ProductionRetrievalRepository,
    VectorHit,
    build_access_filter,
)
from semikb.rag_retrieval.production_service import (
    HydeResult,
    ProductionRetrievalService,
    RetrievalOptions,
)
from semikb.rag_retrieval.rerankers import QianwenReranker, RerankerError


def production_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "demo_mode": False,
        "embedding_dim": 4,
        "retrieval_recall_k": 10,
        "retrieval_min_evidence": 1,
        "retrieval_max_evidence": 5,
        "retrieval_rerank_min_score": 0.35,
        "rerank_api_base_url": "https://rerank.invalid",
        "rerank_api_key": "secret",
        "rerank_model": "qwen3-rerank",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_physical_collection_override_must_match_configured_index() -> None:
    with pytest.raises(ValueError, match="MILVUS_SEARCH_COLLECTION"):
        ProductionRetrievalRepository(
            production_settings(
                milvus_index_version="v4",
                milvus_search_collection="semikb_chunks_v3",
            )
        )

    repository = ProductionRetrievalRepository(
        production_settings(
            milvus_index_version="v4",
            milvus_search_collection="semikb_chunks_v4",
        )
    )
    assert repository._search_collection == "semikb_chunks_v4"


def chunk(
    chunk_id: str,
    text: str,
    *,
    chunk_type: ChunkType = ChunkType.TEXT,
    image_ids: list[str] | None = None,
    product: str = "P-ALPHA",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="CASE-ETCH-03",
        revision="R1",
        chunk_type=chunk_type,
        chunk_text=text,
        title_path=["ETCH-03", "排查"],
        page_or_section="排查",
        lifecycle=DocumentLifecycle.PUBLISHED,
        fab="FAB-01",
        product=product,
        tool_id="ETCH-03",
        chamber="B",
        image_ids=image_ids or [],
    )


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[HybridEmbedding]:
        return [
            HybridEmbedding(dense=[float(index + 1)] * 4, sparse={index + 1: 1.0})
            for index, _ in enumerate(texts)
        ]


class FakeHyde:
    def generate(self, query: str) -> HydeResult:
        assert query
        return HydeResult("假设段落：检查 chamber pressure 和 edge-ring。", "closeai", "luna-test")


class FakeReranker:
    model_name = "reranker-test"

    def score(self, query: str, passages: list[str]) -> list[float]:
        assert query
        score_by_text = {
            "边缘环状": 0.92,
            "图文证据": 0.82,
            "无关": 0.12,
        }
        return [
            next(score for token, score in score_by_text.items() if token in passage)
            for passage in passages
        ]


class FailingReranker:
    model_name = "reranker-test"

    def score(self, query: str, passages: list[str]) -> list[float]:
        raise RerankerError("injected failure")


class CliffReranker:
    model_name = "reranker-test"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.9, 0.3, 0.2]


class FakeRepository:
    def __init__(self) -> None:
        self.chunks = {
            "C1": chunk("C1", "清腔后首片出现边缘环状缺陷。"),
            "C2": chunk(
                "C2",
                "图文证据显示 edge-ring pattern。",
                chunk_type=ChunkType.IMAGE_TEXT,
                image_ids=["IMG-1"],
            ),
            "C3": chunk("C3", "无关 CMP 维护说明。"),
        }
        self.traces: list[RetrievalTrace] = []

    def vector_search(
        self,
        vector: list[float] | dict[int, float],
        *,
        vector_field: str,
        filter_expression: str,
        limit: int,
    ) -> list[VectorHit]:
        assert 'approval_status == "approved"' in filter_expression
        assert limit == 10
        if vector_field == "sparse_vector":
            return [VectorHit("C2", 0.7, 1), VectorHit("C1", 0.6, 2)]
        if isinstance(vector, list) and vector[0] == 2.0:
            return [VectorHit("C2", 0.85, 1), VectorHit("C1", 0.8, 2)]
        return [
            VectorHit("C1", 0.9, 1),
            VectorHit("C2", 0.8, 2),
            VectorHit("C3", 0.2, 3),
        ]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        return {chunk_id: self.chunks[chunk_id] for chunk_id in chunk_ids}

    def save_trace(self, trace: RetrievalTrace) -> RetrievalTrace:
        self.traces.append(trace)
        return trace

    @staticmethod
    def is_accessible(
        item: Chunk,
        actor_scope: ActorScope,
        current: datetime,
        constraints: RetrievalConstraints | None,
    ) -> bool:
        return ProductionRetrievalRepository.is_accessible(
            item,
            actor_scope,
            current,
            constraints,
        )


def test_access_filter_enforces_governance_before_vector_search() -> None:
    actor = ActorScope(
        access_scope_keys=['demo" or lifecycle == "staged'],
        fabs=["FAB-01"],
        products=["P-ALPHA"],
        tool_ids=["ETCH-03"],
    )
    expression, metadata = build_access_filter(
        actor,
        RetrievalConstraints(chamber="B"),
        now=datetime(2026, 8, 12, tzinfo=UTC),
    )

    assert 'approval_status == "approved"' in expression
    assert 'lifecycle == "published"' in expression
    assert 'access_scope_key in ["demo\\" or lifecycle == \\"staged"]' in expression
    assert 'chamber == "B"' in expression
    assert metadata["chamber"] == "B"


def test_defense_in_depth_accepts_pymongo_naive_utc_datetimes() -> None:
    item = chunk("C1", "evidence")
    item.effective_at = datetime(2026, 1, 1)

    assert ProductionRetrievalRepository.is_accessible(
        item,
        ActorScope(),
        datetime(2026, 8, 12, tzinfo=UTC),
        None,
    )


def test_production_pipeline_fuses_routes_reranks_cuts_off_and_returns_image() -> None:
    repository = FakeRepository()
    service = ProductionRetrievalService(
        production_settings(),
        repository=repository,
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        hyde_generator=FakeHyde(),
    )

    evidence, trace = service.search(
        "ETCH-03 Chamber B 首片异常如何排查？",
        ActorScope(),
        constraints=RetrievalConstraints(chamber="B", use_hyde=True),
    )

    assert [item.chunk_id for item in evidence] == ["C1", "C2"]
    assert trace.routes == ["dense", "sparse", "hyde", "rrf", "reranker"]
    assert trace.hyde_query
    assert trace.image_asset_ids == ["IMG-1"]
    assert trace.candidates[0].route_ranks
    assert next(item for item in trace.candidates if item.chunk_id == "C3").exclusion_reason == "below_rerank_threshold"
    assert repository.traces[0].trace_id == trace.trace_id


def test_defense_in_depth_removes_scope_mismatch_even_if_milvus_returns_it() -> None:
    repository = FakeRepository()
    repository.chunks["C3"] = chunk("C3", "P-BETA 无关资料。", product="P-BETA")
    service = ProductionRetrievalService(
        production_settings(hyde_enabled=False),
        repository=repository,
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        hyde_generator=FakeHyde(),
    )

    evidence, trace = service.search("ETCH-03 边缘环状缺陷", ActorScope())

    assert "C3" not in [item.chunk_id for item in evidence]
    assert "C3" not in [item.chunk_id for item in trace.candidates]


def test_reranker_failure_degrades_to_rrf_with_trace_warning() -> None:
    service = ProductionRetrievalService(
        production_settings(hyde_enabled=False),
        repository=FakeRepository(),
        encoder=FakeEncoder(),
        reranker=FailingReranker(),
        hyde_generator=FakeHyde(),
    )

    evidence, trace = service.search("ETCH-03 edge-ring", ActorScope())

    assert evidence
    assert trace.routes[-1] == "rrf_fallback"
    assert trace.warnings == ["reranker_unavailable:RerankerError"]


def test_dynamic_cutoff_stops_at_a_rerank_score_cliff() -> None:
    service = ProductionRetrievalService(
        production_settings(hyde_enabled=False, retrieval_rerank_min_score=0.1),
        repository=FakeRepository(),
        encoder=FakeEncoder(),
        reranker=CliffReranker(),
        hyde_generator=FakeHyde(),
    )

    evidence, trace = service.search("ETCH-03 edge-ring", ActorScope())

    assert [item.chunk_id for item in evidence] == ["C1"]
    assert trace.cutoff_reason == "rerank_score_cliff"


def test_protected_evidence_replaces_lower_priority_non_protected_candidate() -> None:
    service = ProductionRetrievalService(
        production_settings(retrieval_max_evidence=2),
        repository=FakeRepository(),
        encoder=FakeEncoder(),
        reranker=FakeReranker(),
        hyde_generator=FakeHyde(),
    )
    candidates = [
        RetrievalCandidate(
            chunk_id=chunk_id,
            document_id="DOC",
            revision="R1",
            title="title",
            page_or_section="section",
            routes=["dense"],
            dense_score=score,
            sparse_score=0,
            rrf_score=score,
            rerank_score=score,
            protected_evidence=protected,
        )
        for chunk_id, score, protected in (
            ("C1", 0.9, False),
            ("C2", 0.8, False),
            ("SOP", 0.4, True),
        )
    ]

    selected, _ = service._select_candidates(
        candidates,
        top_k=2,
        apply_threshold=True,
    )

    assert [item.chunk_id for item in selected] == ["C1", "SOP"]


def test_dense_only_baseline_does_not_call_hyde_or_reranker() -> None:
    service = ProductionRetrievalService(
        production_settings(),
        repository=FakeRepository(),
        encoder=FakeEncoder(),
        reranker=FailingReranker(),
        hyde_generator=FakeHyde(),
    )

    _, trace = service.search(
        "异常原因",
        ActorScope(),
        options=RetrievalOptions(dense=True, sparse=False, rerank=False, hyde=False),
    )

    assert trace.routes == ["dense", "rrf", "rrf_only"]
    assert not trace.warnings


def test_qianwen_reranker_restores_scores_to_original_document_order() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.2},
                        {"index": 0, "relevance_score": 0.9},
                    ]
                }
            },
        )

    reranker = QianwenReranker(
        production_settings(),
        transport=httpx.MockTransport(handler),
    )
    scores = reranker.score("query", ["doc-a", "doc-b"])

    assert scores == [0.9, 0.2]
    assert captured["model"] == "qwen3-rerank"
    assert captured["parameters"] == {"return_documents": False, "top_n": 2}
