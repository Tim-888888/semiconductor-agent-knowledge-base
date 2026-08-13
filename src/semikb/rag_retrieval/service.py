"""Deterministic demo hybrid retrieval with production-compatible trace output."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from time import perf_counter

from semikb.contracts.models import (
    ActorScope,
    Chunk,
    RetrievalCandidate,
    RetrievalConstraints,
    RetrievalTrace,
)
from semikb.storage.memory import DemoStore

IDENTIFIER_PATTERN = re.compile(r"[a-z]+-\d+(?:\.\d+)?|v\d+(?:\.\d+)?|[a-z]+\d*", re.IGNORECASE)
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(value: str) -> set[str]:
    """Create identifier tokens plus short Chinese n-grams for a dependency-free demo."""

    value = value.lower()
    tokens = set(IDENTIFIER_PATTERN.findall(value))
    for phrase in CHINESE_PATTERN.findall(value):
        if len(phrase) <= 2:
            tokens.add(phrase)
            continue
        tokens.update(phrase[index : index + 2] for index in range(len(phrase) - 1))
        tokens.add(phrase)
    return tokens


class RetrievalService:
    """A replaceable service: deterministic demo scoring or configured production retrieval."""

    def __init__(self, store: DemoStore) -> None:
        self.store = store

    def search(
        self,
        query: str,
        actor_scope: ActorScope,
        *,
        top_k: int = 5,
        thread_id: str | None = None,
        constraints: RetrievalConstraints | None = None,
        options: object | None = None,
    ) -> tuple[list[Chunk], RetrievalTrace]:
        started = perf_counter()
        query_tokens = tokenize(query)
        accessible = self.store.list_published_chunks(actor_scope)
        if constraints:
            accessible = [
                chunk
                for chunk in accessible
                if all(
                    not getattr(constraints, field)
                    or getattr(chunk, field) == getattr(constraints, field)
                    for field in (
                        "fab",
                        "product",
                        "process_layer",
                        "tool_id",
                        "chamber",
                        "recipe_id",
                        "recipe_version",
                    )
                )
            ]
        scored = [self._score(chunk, query_tokens) for chunk in accessible]
        dense_ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        sparse_ranked = sorted(scored, key=lambda item: item[2], reverse=True)
        dense_positions = {chunk.chunk_id: index + 1 for index, (chunk, _, _) in enumerate(dense_ranked)}
        sparse_positions = {chunk.chunk_id: index + 1 for index, (chunk, _, _) in enumerate(sparse_ranked)}

        candidates: list[RetrievalCandidate] = []
        for chunk, dense_score, sparse_score in scored:
            rrf_score = 1 / (60 + dense_positions[chunk.chunk_id]) + 1 / (60 + sparse_positions[chunk.chunk_id])
            exact_boost = self._exact_constraint_boost(chunk, query)
            rerank_score = dense_score * 0.65 + sparse_score * 0.35 + exact_boost
            protected = self._is_protected_evidence(chunk, query)
            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    revision=chunk.revision,
                    title=" > ".join(chunk.title_path),
                    page_or_section=chunk.page_or_section,
                    routes=["dense", "sparse"],
                    dense_score=round(dense_score, 5),
                    sparse_score=round(sparse_score, 5),
                    rrf_score=round(rrf_score, 5),
                    rerank_score=round(rerank_score, 5),
                    protected_evidence=protected,
                )
            )

        candidates.sort(key=lambda candidate: candidate.rerank_score, reverse=True)
        selected, cutoff_reason = self._select_candidates(candidates, top_k)
        selected_ids = {candidate.chunk_id for candidate in selected}
        for candidate in candidates:
            candidate.selected = candidate.chunk_id in selected_ids
            if not candidate.selected:
                candidate.exclusion_reason = "dynamic_cutoff"

        image_asset_ids = [
            image_id
            for chunk_id in selected_ids
            for image_id in (self.store.get_chunk(chunk_id).image_ids if self.store.get_chunk(chunk_id) else [])
        ]
        trace = RetrievalTrace(
            thread_id=thread_id,
            actor_user_id=actor_scope.user_id,
            access_scope_keys=actor_scope.access_scope_keys,
            original_query=query,
            metadata_filters={
                "fabs": actor_scope.fabs,
                "products": actor_scope.products,
                "tool_ids": actor_scope.tool_ids,
                "access_scope_keys": actor_scope.access_scope_keys,
                "approval_status": "approved",
                "lifecycle": "published",
                **(
                    constraints.model_dump(mode="json", exclude_none=True)
                    if constraints
                    else {}
                ),
            },
            routes=["dense", "sparse", "rrf", "reranker"],
            candidates=candidates,
            cutoff_reason=cutoff_reason,
            final_evidence_ids=[candidate.chunk_id for candidate in selected],
            image_asset_ids=image_asset_ids,
            timings_ms={"retrieval": round((perf_counter() - started) * 1000, 2)},
        )
        self.store.save_trace(trace)
        return [self.store.get_chunk(candidate.chunk_id) for candidate in selected if self.store.get_chunk(candidate.chunk_id)], trace

    def get_trace(
        self,
        trace_id: str,
        actor_scope: ActorScope | None = None,
    ) -> RetrievalTrace | None:
        return self.store.get_trace(trace_id, actor_scope)

    def reuse_trace_evidence(
        self,
        trace_id: str,
        actor_scope: ActorScope,
        *,
        constraints: RetrievalConstraints | None = None,
    ) -> tuple[list[Chunk], RetrievalTrace] | None:
        trace = self.store.get_trace(trace_id, actor_scope)
        if trace is None or not trace.final_evidence_ids:
            return None
        accessible = {
            chunk.chunk_id: chunk for chunk in self.store.list_published_chunks(actor_scope)
        }
        selected = []
        for chunk_id in trace.final_evidence_ids:
            chunk = accessible.get(chunk_id)
            if chunk is None:
                return None
            if constraints and any(
                getattr(constraints, field) and getattr(chunk, field) != getattr(constraints, field)
                for field in (
                    "fab",
                    "product",
                    "process_layer",
                    "tool_id",
                    "chamber",
                    "recipe_id",
                    "recipe_version",
                )
            ):
                return None
            selected.append(chunk)
        return selected, trace

    def list_traces(self, actor_scope: ActorScope | None = None) -> list[RetrievalTrace]:
        return self.store.list_traces(actor_scope)

    def save_trace(self, trace: RetrievalTrace) -> RetrievalTrace:
        return self.store.save_trace(trace)

    @staticmethod
    def _score(chunk: Chunk, query_tokens: set[str]) -> tuple[Chunk, float, float]:
        chunk_tokens = tokenize(f"{' '.join(chunk.title_path)} {chunk.chunk_text}")
        overlap = query_tokens.intersection(chunk_tokens)
        dense_score = len(overlap) / max(len(query_tokens.union(chunk_tokens)), 1)
        sparse_score = sum(1.0 for token in overlap if IDENTIFIER_PATTERN.fullmatch(token))
        sparse_score += len(overlap) / max(len(query_tokens), 1)
        if chunk.chunk_type.value == "image_text" and {"晶圆", "缺陷", "图"}.intersection(query_tokens):
            dense_score += 0.08
        return chunk, dense_score, sparse_score

    @staticmethod
    def _exact_constraint_boost(chunk: Chunk, query: str) -> float:
        normal = query.lower()
        score = 0.0
        if chunk.tool_id and chunk.tool_id.lower() in normal:
            score += 0.25
        if chunk.chamber and f"chamber {chunk.chamber.lower()}" in normal:
            score += 0.15
        if chunk.recipe_version and chunk.recipe_version.lower() in normal:
            score += 0.1
        return score

    @staticmethod
    def _is_protected_evidence(chunk: Chunk, query: str) -> bool:
        normal = query.lower()
        return bool(
            chunk.document_id.startswith("SOP-")
            and chunk.tool_id
            and chunk.tool_id.lower() in normal
            and chunk.chamber
            and f"chamber {chunk.chamber.lower()}" in normal
        )

    @staticmethod
    def _select_candidates(candidates: list[RetrievalCandidate], top_k: int) -> tuple[list[RetrievalCandidate], str]:
        if not candidates:
            return [], "no_accessible_candidates"
        max_docs = min(max(top_k, 3), 8)
        selected = candidates[:max_docs]
        if len(candidates) <= max_docs:
            return selected, "candidate_count_within_limit"
        min_docs = min(3, len(selected))
        for index in range(min_docs, len(selected)):
            previous = selected[index - 1].rerank_score
            current = selected[index].rerank_score
            if previous > 0 and (previous - current) / previous >= 0.45:
                return selected[:index], "rerank_score_cliff"
        return selected, "top_k"

    def asset_access(self, image_id: str, actor_scope: ActorScope) -> dict[str, str]:
        asset = self.store.get_image(image_id)
        if asset is None:
            raise KeyError(image_id)
        document = self.store.get_document(asset.document_id, asset.revision)
        if document is None:
            raise KeyError(image_id)
        probe = Chunk(
            chunk_id=f"asset-{asset.image_id}",
            document_id=asset.document_id,
            revision=asset.revision,
            chunk_text=asset.caption,
            title_path=[asset.image_type],
            page_or_section=asset.source_page,
            approval_status=asset.approval_status,
            lifecycle=asset.lifecycle,
            effective_at=asset.effective_at,
            expires_at=asset.expires_at,
            access_scope_key=asset.access_scope_key,
            fab=document.fab,
            product=document.product,
            tool_id=document.tool_id,
        )
        if not self.store._is_accessible(probe, actor_scope, datetime.now(UTC)):
            raise PermissionError(image_id)
        expires_at = datetime.now(UTC) + timedelta(minutes=5)
        return {
            "image_id": image_id,
            "url": f"/api/v1/assets/{image_id}/preview",
            "expires_at": expires_at.isoformat(),
            "object_key": asset.object_ref.object_key,
        }
