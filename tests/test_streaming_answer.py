from __future__ import annotations

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
