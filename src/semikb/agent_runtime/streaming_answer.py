"""Incrementally validate model answer units before exposing them to users."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from semikb.contracts.models import AgentAnswer, AnswerClaim, EvidenceLedgerEntry


def format_answer(answer: AgentAnswer) -> str:
    lines = ["基于当前有效且有权限访问的受控证据："]
    if answer.facts:
        lines.append("\n已知事实")
        lines.extend(
            f"- {claim.text} [{', '.join(claim.citation_ids)}]" for claim in answer.facts
        )
    if answer.hypotheses:
        lines.append("\n待验证假设")
        lines.extend(
            f"- {claim.text} [{', '.join(claim.citation_ids)}]"
            for claim in answer.hypotheses
        )
    if answer.unknowns:
        lines.append("\n仍不确定")
        lines.extend(f"- {item}" for item in answer.unknowns)
    if answer.next_actions:
        lines.append("\n建议下一步")
        lines.extend(f"- {item}" for item in answer.next_actions)
    lines.append(f"\n置信度：{answer.confidence}")
    return "\n".join(lines)


class StreamingAnswerAssembler:
    """Parse JSON objects from arbitrary network chunks and emit verified text growth."""

    _ORDER = {
        "fact": 0,
        "hypothesis": 1,
        "unknown": 2,
        "next_action": 3,
        "confidence": 4,
    }

    def __init__(
        self,
        ledger: list[EvidenceLedgerEntry],
        emit_delta: Callable[[str], None],
    ) -> None:
        self._ledger = ledger
        self._emit_delta = emit_delta
        self._buffer = ""
        self._decoder = json.JSONDecoder()
        self._facts: list[AnswerClaim] = []
        self._hypotheses: list[AnswerClaim] = []
        self._unknowns: list[str] = []
        self._next_actions: list[str] = []
        self._confidence: str | None = None
        self._last_order = -1
        self._rendered = ""
        self._processed_units = 0
        self.warnings: list[str] = []

        self._valid_ids = {item.evidence_id for item in ledger}
        self._citation_aliases = {
            alias: item.evidence_id
            for item in ledger
            for alias in (item.chunk_id, item.document_id)
            if alias
        }
        self._source_by_id = {item.evidence_id: item.source_type for item in ledger}
        self._has_internal = any(item.source_type == "internal_controlled" for item in ledger)

    def feed(self, delta: str) -> None:
        if not delta:
            return
        self._buffer += delta
        self._consume_available(final=False)

    def finish(self) -> AgentAnswer:
        self._consume_available(final=True)
        if self._processed_units == 0:
            raise ValueError("streaming answer contained no valid answer units")
        if self._confidence is None:
            self.warnings.append("confidence_defaulted_to_low")
            self._confidence = "low"
            self._emit_render_growth()
        return self.answer

    @property
    def answer(self) -> AgentAnswer:
        return AgentAnswer(
            facts=self._facts,
            hypotheses=self._hypotheses,
            unknowns=self._unknowns,
            next_actions=self._next_actions,
            confidence=self._confidence or "low",
        )

    @property
    def rendered_text(self) -> str:
        return self._rendered

    def _consume_available(self, *, final: bool) -> None:
        while True:
            self._strip_prefix_noise()
            if not self._buffer:
                return
            try:
                payload, end = self._decoder.raw_decode(self._buffer)
            except json.JSONDecodeError as exc:
                if final:
                    remainder = self._buffer.strip()
                    if remainder not in {"", "```"}:
                        raise ValueError("streaming answer ended with invalid JSON") from exc
                    self._buffer = ""
                return
            self._buffer = self._buffer[end:]
            self._process_unit(payload)

    def _strip_prefix_noise(self) -> None:
        self._buffer = self._buffer.lstrip()
        for prefix in ("```json", "```JSON", "```"):
            if self._buffer.startswith(prefix):
                self._buffer = self._buffer[len(prefix) :].lstrip()
                break

    def _process_unit(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            self.warnings.append("non_object_answer_unit_removed")
            return
        unit_type = str(payload.get("type", "")).strip().lower()
        order = self._ORDER.get(unit_type)
        if order is None:
            self.warnings.append("unknown_answer_unit_removed")
            return
        if order < self._last_order:
            self.warnings.append("out_of_order_answer_unit_removed")
            return
        if self._confidence is not None:
            self.warnings.append("answer_unit_after_confidence_removed")
            return

        changed = False
        if unit_type in {"fact", "hypothesis"}:
            claim = self._validated_claim(payload, fact=unit_type == "fact")
            if claim is not None:
                target = self._facts if unit_type == "fact" else self._hypotheses
                target.append(claim)
                changed = True
        elif unit_type in {"unknown", "next_action"}:
            text = self._clean_text(payload.get("text"))
            if text:
                target = self._unknowns if unit_type == "unknown" else self._next_actions
                target.append(text)
                changed = True
        else:
            confidence = str(payload.get("value", payload.get("confidence", ""))).lower()
            if confidence in {"low", "medium", "high"}:
                self._confidence = confidence
                changed = True
            else:
                self.warnings.append("invalid_confidence_defaulted_to_low")
                self._confidence = "low"
                changed = True

        self._last_order = max(self._last_order, order)
        self._processed_units += 1
        if changed:
            self._emit_render_growth()

    def _validated_claim(self, payload: dict[str, Any], *, fact: bool) -> AnswerClaim | None:
        text = self._clean_text(payload.get("text"))
        if not text:
            self.warnings.append("empty_claim_removed")
            return None
        citations = payload.get("citation_ids", payload.get("citations", []))
        if isinstance(citations, str):
            citations = [citations]
        if not isinstance(citations, list):
            citations = []
        normalized = [
            self._citation_aliases.get(str(item), str(item))
            for item in citations
            if self._citation_aliases.get(str(item), str(item)) in self._valid_ids
        ]
        normalized = list(dict.fromkeys(normalized))
        if fact and not normalized:
            self.warnings.append("fact_removed_without_valid_citation")
            return None
        if fact and self._has_internal and all(
            self._source_by_id[citation] == "external" for citation in normalized
        ):
            self.warnings.append("external_only_fact_removed_when_internal_evidence_exists")
            return None
        if not fact:
            text = text.replace("根因是", "待验证假设是")
        return AnswerClaim(text=text, citation_ids=normalized)

    @staticmethod
    def _clean_text(value: Any) -> str:
        return str(value or "").strip()[:2000]

    def _emit_render_growth(self) -> None:
        rendered = format_answer(self.answer)
        if self._confidence is None:
            rendered = rendered.rsplit("\n置信度：", maxsplit=1)[0]
        if not rendered.startswith(self._rendered):
            raise ValueError("answer units cannot be rendered incrementally")
        delta = rendered[len(self._rendered) :]
        self._rendered = rendered
        if delta:
            self._emit_delta(delta)
