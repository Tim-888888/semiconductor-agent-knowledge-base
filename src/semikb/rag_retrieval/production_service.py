"""Production hybrid retrieval over governed Milvus and MongoDB records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

from semikb.agent_runtime.llm_gateway import OpenAICompatibleLLMGateway
from semikb.config import Settings
from semikb.contracts.models import (
    ActorScope,
    Chunk,
    RetrievalCandidate,
    RetrievalConstraints,
    RetrievalTrace,
)
from semikb.rag_retrieval.encoders import BgeM3Encoder, HybridEncoder
from semikb.rag_retrieval.production_repository import (
    ProductionRetrievalRepository,
    VectorHit,
    build_access_filter,
)
from semikb.rag_retrieval.rerankers import Reranker, RerankerError, create_reranker


@dataclass(frozen=True, slots=True)
class HydeResult:
    text: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    dense: bool = True
    sparse: bool = True
    rerank: bool = True
    hyde: bool | None = None

    def __post_init__(self) -> None:
        if not self.dense and not self.sparse:
            raise ValueError("At least one of dense or sparse retrieval must be enabled.")


class HydeGenerator:
    """Generate a hypothetical retrieval passage, never a user-facing answer."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._gateway = OpenAICompatibleLLMGateway(settings)

    def generate(self, query: str) -> HydeResult:
        result = self._gateway.complete_sync(
            [
                {
                    "role": "system",
                    "content": (
                        "你是半导体知识检索查询扩展器。根据问题写一段可能出现在受控 SOP、"
                        "Recipe、FDC 规则或历史 Case 中的简短技术段落。保留问题中的 Tool、"
                        "Chamber、Recipe、Lot 和参数标识。不要声称已找到真实证据，不要输出引用，"
                        "只输出用于向量检索的假设段落。"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_output_tokens=self._settings.hyde_max_output_tokens,
            allow_fallback=True,
        )
        return HydeResult(
            text=result.content.strip(),
            provider=result.provider,
            model=result.reported_model,
        )


class ProductionRetrievalService:
    """Dense + sparse + optional HyDE recall, RRF, rerank, and governed assets."""

    def __init__(
        self,
        settings: Settings,
        *,
        repository: ProductionRetrievalRepository | None = None,
        encoder: HybridEncoder | None = None,
        reranker: Reranker | None = None,
        hyde_generator: HydeGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository or ProductionRetrievalRepository(settings)
        self.encoder = encoder or BgeM3Encoder(settings)
        self.reranker = reranker or create_reranker(settings)
        self.hyde_generator = hyde_generator or HydeGenerator(settings)

    def search(
        self,
        query: str,
        actor_scope: ActorScope,
        *,
        top_k: int = 5,
        thread_id: str | None = None,
        constraints: RetrievalConstraints | None = None,
        options: RetrievalOptions | None = None,
    ) -> tuple[list[Chunk], RetrievalTrace]:
        selected_options = options or RetrievalOptions()
        started = perf_counter()
        timings: dict[str, float] = {}
        warnings: list[str] = []
        component_versions = {
            "embedding": "bge-m3",
            "embedding_dim": str(self.settings.embedding_dim),
            "reranker": self.reranker.model_name,
            "index_version": self.settings.milvus_index_version,
        }
        filter_expression, metadata_filters = build_access_filter(
            actor_scope,
            constraints,
            now=datetime.now(UTC),
        )

        hyde_result: HydeResult | None = None
        if self._should_use_hyde(query, constraints, selected_options):
            stage = perf_counter()
            try:
                hyde_result = self.hyde_generator.generate(query)
                component_versions["hyde_provider"] = hyde_result.provider
                component_versions["hyde_model"] = hyde_result.model
            except Exception as exc:
                warnings.append(f"hyde_unavailable:{type(exc).__name__}")
            timings["hyde"] = round((perf_counter() - stage) * 1000, 2)

        texts = [query]
        if hyde_result:
            texts.append(hyde_result.text)
        stage = perf_counter()
        embeddings = self.encoder.encode(texts)
        timings["embedding"] = round((perf_counter() - stage) * 1000, 2)

        stage = perf_counter()
        route_hits: dict[str, list[VectorHit]] = {}
        if selected_options.dense:
            route_hits["dense"] = self.repository.vector_search(
                embeddings[0].dense,
                vector_field="dense_vector",
                filter_expression=filter_expression,
                limit=self.settings.retrieval_recall_k,
            )
        if selected_options.sparse:
            route_hits["sparse"] = self.repository.vector_search(
                embeddings[0].sparse,
                vector_field="sparse_vector",
                filter_expression=filter_expression,
                limit=self.settings.retrieval_recall_k,
            )
        if hyde_result and selected_options.dense:
            route_hits["hyde"] = self.repository.vector_search(
                embeddings[1].dense,
                vector_field="dense_vector",
                filter_expression=filter_expression,
                limit=self.settings.retrieval_recall_k,
            )
        timings["recall"] = round((perf_counter() - stage) * 1000, 2)

        candidate_ids = list(
            dict.fromkeys(hit.chunk_id for hits in route_hits.values() for hit in hits)
        )
        chunks_by_id = self.repository.get_chunks(candidate_ids)
        current = datetime.now(UTC)
        chunks_by_id = {
            chunk_id: chunk
            for chunk_id, chunk in chunks_by_id.items()
            if self.repository.is_accessible(chunk, actor_scope, current, constraints)
        }
        candidates = self._build_candidates(query, route_hits, chunks_by_id)

        stage = perf_counter()
        reranker_applied = False
        reranker_failed = False
        if candidates and selected_options.rerank:
            passages = [
                f"{' > '.join(chunks_by_id[item.chunk_id].title_path)}\n"
                f"{chunks_by_id[item.chunk_id].chunk_text}"
                for item in candidates
            ]
            try:
                scores = self.reranker.score(query, passages)
                for candidate, score in zip(candidates, scores, strict=True):
                    candidate.rerank_score = round(float(score), 6)
                reranker_applied = True
            except (RerankerError, RuntimeError, ValueError) as exc:
                reranker_failed = True
                warnings.append(f"reranker_unavailable:{type(exc).__name__}")
                max_rrf = max((candidate.rrf_score for candidate in candidates), default=1.0)
                for candidate in candidates:
                    candidate.rerank_score = round(candidate.rrf_score / max_rrf, 6)
        elif candidates:
            max_rrf = max((candidate.rrf_score for candidate in candidates), default=1.0)
            for candidate in candidates:
                candidate.rerank_score = round(candidate.rrf_score / max_rrf, 6)
        timings["rerank"] = round((perf_counter() - stage) * 1000, 2)

        candidates.sort(key=lambda item: item.rerank_score, reverse=True)
        selected, cutoff_reason = self._select_candidates(
            candidates,
            top_k=top_k,
            apply_threshold=reranker_applied,
        )
        selected_ids = {candidate.chunk_id for candidate in selected}
        for candidate in candidates:
            if candidate.chunk_id in selected_ids:
                candidate.selected = True
                candidate.context_selection_reason = (
                    "protected_evidence"
                    if candidate.protected_evidence
                    else "rerank_selected"
                )
            elif reranker_applied and candidate.rerank_score < self.settings.retrieval_rerank_min_score:
                candidate.exclusion_reason = "below_rerank_threshold"
            else:
                candidate.exclusion_reason = "dynamic_cutoff"

        image_asset_ids = list(
            dict.fromkeys(
                image_id
                for candidate in selected
                for image_id in chunks_by_id[candidate.chunk_id].image_ids
            )
        )
        executed_routes = [route for route, hits in route_hits.items() if hits]
        timings["total"] = round((perf_counter() - started) * 1000, 2)
        if reranker_applied:
            final_stage = "reranker"
        elif reranker_failed:
            final_stage = "rrf_fallback"
        elif not candidates:
            final_stage = "no_candidates"
        else:
            final_stage = "rrf_only"
        trace = RetrievalTrace(
            thread_id=thread_id,
            actor_user_id=actor_scope.user_id,
            access_scope_keys=actor_scope.access_scope_keys,
            original_query=query,
            hyde_query=hyde_result.text if hyde_result else None,
            metadata_filters=metadata_filters,
            routes=[*executed_routes, "rrf", final_stage],
            candidates=candidates,
            cutoff_reason=cutoff_reason,
            final_evidence_ids=[candidate.chunk_id for candidate in selected],
            image_asset_ids=image_asset_ids,
            component_versions=component_versions,
            warnings=warnings,
            timings_ms=timings,
        )
        self.repository.save_trace(trace)
        return [chunks_by_id[candidate.chunk_id] for candidate in selected], trace

    def asset_access(self, image_id: str, actor_scope: ActorScope) -> dict[str, str]:
        return self.repository.asset_access(image_id, actor_scope)

    def get_trace(
        self,
        trace_id: str,
        actor_scope: ActorScope | None = None,
    ) -> RetrievalTrace | None:
        return self.repository.get_trace(trace_id, actor_scope)

    def list_traces(self, actor_scope: ActorScope | None = None) -> list[RetrievalTrace]:
        return self.repository.list_traces(actor_scope)

    def save_trace(self, trace: RetrievalTrace) -> RetrievalTrace:
        return self.repository.save_trace(trace)

    def _build_candidates(
        self,
        query: str,
        route_hits: dict[str, list[VectorHit]],
        chunks_by_id: dict[str, Chunk],
    ) -> list[RetrievalCandidate]:
        hits_by_route = {
            route: {hit.chunk_id: hit for hit in hits}
            for route, hits in route_hits.items()
        }
        candidates: list[RetrievalCandidate] = []
        for chunk_id, chunk in chunks_by_id.items():
            routes = [route for route, hits in hits_by_route.items() if chunk_id in hits]
            route_ranks = {route: hits_by_route[route][chunk_id].rank for route in routes}
            rrf_score = sum(
                1 / (self.settings.retrieval_rrf_k + rank)
                for rank in route_ranks.values()
            )
            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    revision=chunk.revision,
                    title=" > ".join(chunk.title_path),
                    page_or_section=chunk.page_or_section,
                    routes=routes,
                    dense_score=round(self._route_score(hits_by_route, "dense", chunk_id), 6),
                    sparse_score=round(self._route_score(hits_by_route, "sparse", chunk_id), 6),
                    hyde_score=round(self._route_score(hits_by_route, "hyde", chunk_id), 6),
                    rrf_score=round(rrf_score, 8),
                    rerank_score=0.0,
                    route_ranks=route_ranks,
                    protected_evidence=self._is_protected_evidence(chunk, query),
                )
            )
        return candidates

    @staticmethod
    def _route_score(
        hits_by_route: dict[str, dict[str, VectorHit]],
        route: str,
        chunk_id: str,
    ) -> float:
        hit = hits_by_route.get(route, {}).get(chunk_id)
        return hit.score if hit else 0.0

    def _select_candidates(
        self,
        candidates: list[RetrievalCandidate],
        *,
        top_k: int,
        apply_threshold: bool,
    ) -> tuple[list[RetrievalCandidate], str]:
        if not candidates:
            return [], "no_accessible_candidates"
        eligible = [
            candidate
            for candidate in candidates
            if not apply_threshold
            or candidate.rerank_score >= self.settings.retrieval_rerank_min_score
            or candidate.protected_evidence
        ]
        if not eligible:
            return [], "all_below_rerank_threshold"
        max_docs = min(top_k, self.settings.retrieval_max_evidence, len(eligible))
        selected = eligible[:max_docs]
        protected = [item for item in eligible if item.protected_evidence]
        selected = self._retain_protected(selected, protected)

        min_docs = min(self.settings.retrieval_min_evidence, len(selected))
        for index in range(min_docs, len(selected)):
            previous = selected[index - 1].rerank_score
            current = selected[index].rerank_score
            if previous > 0 and (previous - current) / previous >= self.settings.retrieval_score_cliff_ratio:
                kept = selected[:index]
                return self._retain_protected(kept, protected), "rerank_score_cliff"
        if len(eligible) > len(selected):
            return selected, "top_k"
        return selected, "candidate_count_within_limit"

    def _retain_protected(
        self,
        selected: list[RetrievalCandidate],
        protected: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        combined = list(selected)
        for item in protected:
            if item not in combined:
                combined.append(item)
        combined.sort(key=lambda item: item.rerank_score, reverse=True)
        while len(combined) > self.settings.retrieval_max_evidence:
            removable = next(
                (item for item in reversed(combined) if not item.protected_evidence),
                None,
            )
            if removable is None:
                break
            combined.remove(removable)
        return combined

    def _should_use_hyde(
        self,
        query: str,
        constraints: RetrievalConstraints | None,
        options: RetrievalOptions,
    ) -> bool:
        if not options.dense:
            return False
        if options.hyde is not None:
            return options.hyde
        if constraints and constraints.use_hyde is not None:
            return constraints.use_hyde
        if not self.settings.hyde_enabled:
            return False
        normalized = query.lower()
        diagnostic_terms = (
            "为什么",
            "原因",
            "根因",
            "异常",
            "怎么",
            "如何",
            "排查",
            "why",
            "cause",
            "troubleshoot",
        )
        return any(term in normalized for term in diagnostic_terms)

    @staticmethod
    def _is_protected_evidence(chunk: Chunk, query: str) -> bool:
        normalized = query.lower()
        return bool(
            chunk.document_id.upper().startswith("SOP-")
            and chunk.tool_id
            and chunk.tool_id.lower() in normalized
            and (not chunk.chamber or f"chamber {chunk.chamber.lower()}" in normalized)
        )
