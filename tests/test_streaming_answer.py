from __future__ import annotations

import json

import pytest

from semikb.agent_runtime.streaming_answer import StreamingAnswerAssembler, format_answer
from semikb.contracts.models import EvidenceLedgerEntry


def test_streaming_answer_emits_only_complete_verified_units() -> None:
    deltas: list[str] = []
    assembler = StreamingAnswerAssembler(
        [
            EvidenceLedgerEntry(
                evidence_id="chunk:approved-1",
                source_type="internal_controlled",
                content="approved",
            ),
            EvidenceLedgerEntry(
                evidence_id="external:web-1",
                source_type="external",
                content="external",
            ),
        ],
        deltas.append,
    )

    assembler.feed('{"type":"fact","text":"受控事实","citation_ids":["chunk:')
    assert deltas == []
    assembler.feed('approved-1"]}\n')
    assembler.feed(
        '{"type":"fact","text":"外部单独结论","citation_ids":["external:web-1"]}\n'
        '{"type":"hypothesis","text":"根因是压力波动","citation_ids":["chunk:approved-1"]}\n'
        '{"type":"next_action","text":"复核 FDC"}\n'
        '{"type":"confidence","value":"medium"}'
    )
    answer = assembler.finish()

    assert [claim.text for claim in answer.facts] == ["受控事实"]
    assert answer.hypotheses[0].text == "待验证假设是压力波动"
    assert "external_only_fact_removed_when_internal_evidence_exists" in assembler.warnings
    assert "".join(deltas) == format_answer(answer)


def test_streaming_answer_rejects_incomplete_json() -> None:
    assembler = StreamingAnswerAssembler([], lambda _: None)
    assembler.feed('{"type":"unknown","text":"incomplete"')

    with pytest.raises(ValueError, match="invalid JSON"):
        assembler.finish()


def test_streaming_answer_adds_cited_fact_before_non_fact_units() -> None:
    deltas: list[str] = []
    ledger = [
        EvidenceLedgerEntry(
            evidence_id="chunk:approved-1",
            source_type="internal_controlled",
            content="受控 SOP 要求复核 RF match。",
        )
    ]
    assembler = StreamingAnswerAssembler(ledger, deltas.append)

    assembler.feed(
        json.dumps(
            {
                "type": "fact",
                "text": "模型给出了无效引用",
                "citation_ids": ["chunk:missing"],
            },
            ensure_ascii=False,
        )
    )
    assembler.feed(
        json.dumps(
            {"type": "next_action", "text": "继续核对受控证据"},
            ensure_ascii=False,
        )
    )
    assembler.feed(json.dumps({"type": "confidence", "value": "medium"}))
    answer = assembler.finish()

    assert answer.facts[0].citation_ids == [ledger[0].evidence_id]
    assert answer.facts[0].text == ledger[0].content
    assert "deterministic_fact_added_without_valid_model_fact" in assembler.warnings
    assert "".join(deltas) == format_answer(answer)


def test_streaming_answer_rejects_full_dataset_total_from_truncated_preview() -> None:
    deltas: list[str] = []
    ledger = [
        EvidenceLedgerEntry(
            evidence_id="chunk:dataset-profile",
            source_type="internal_controlled",
            content=(
                "Observed rows: 200\nColumn count: 590\n"
                "Sample truncated: true\nColumns truncated: true"
            ),
        )
    ]
    assembler = StreamingAnswerAssembler(ledger, deltas.append)

    assembler.feed(
        json.dumps(
            {
                "type": "fact",
                "text": "该数据集包含 200 个样本，共有 590 列。",
                "citation_ids": ["chunk:dataset-profile"],
            },
            ensure_ascii=False,
        )
    )
    assembler.feed(json.dumps({"type": "confidence", "value": "medium"}))
    answer = assembler.finish()

    assert answer.facts[0].text == ledger[0].content
    assert "bounded_preview_total_claim_removed" in assembler.warnings
    assert "deterministic_fact_added_without_valid_model_fact" in assembler.warnings
    assert "".join(deltas) == format_answer(answer)


def test_streaming_answer_keeps_explicitly_qualified_preview_count() -> None:
    ledger = [
        EvidenceLedgerEntry(
            evidence_id="chunk:dataset-profile",
            source_type="internal_controlled",
            content="Observed preview rows: 200\nSample truncated: true",
        )
    ]
    assembler = StreamingAnswerAssembler(ledger, lambda _: None)

    assembler.feed(
        json.dumps(
            {
                "type": "fact",
                "text": "当前预览观测到 200 行，且证据标记为截断样本。",
                "citation_ids": ["chunk:dataset-profile"],
            },
            ensure_ascii=False,
        )
    )
    assembler.feed(json.dumps({"type": "confidence", "value": "medium"}))
    answer = assembler.finish()

    assert answer.facts[0].text == "当前预览观测到 200 行，且证据标记为截断样本。"
    assert "bounded_preview_total_claim_removed" not in assembler.warnings
