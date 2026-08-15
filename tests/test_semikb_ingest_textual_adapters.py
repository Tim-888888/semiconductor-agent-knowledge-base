from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from semikb_ingest import ChunkType, SourceFormat, build_dispatcher
from semikb_ingest.assets import ProcessPayloadStore

_CONTRACT = json.loads(
    Path("docs/evidence/t9-4-4-1/semikb-ingest-contract-v1.schema.json").read_text(encoding="utf-8")
)


def _assert_frozen_contract(parsed) -> None:
    Draft202012Validator(_CONTRACT).validate(parsed.model_dump(mode="json"))


def test_markdown_preserves_heading_code_list_and_table() -> None:
    content = """# ETCH-03 SOP

## 首片异常处理

- 暂停后续 lot 放行
- 复核 chamber pressure

| 参数 | 单位 | 上限 |
| --- | --- | --- |
| Pressure | Pa | 12 |

```python
alarm = pressure > 12
```
""".encode()

    parsed = build_dispatcher(ProcessPayloadStore()).parse(
        "etch_sop.md",
        content,
        declared_media_type="text/markdown",
        correlation_id="md-1",
    )

    assert parsed.source_format is SourceFormat.MARKDOWN
    assert parsed.detected_title == "ETCH-03 SOP"
    assert len(parsed.tables) == 1
    assert parsed.tables[0].headers == ("参数", "单位", "上限")
    assert any(chunk.chunk_type is ChunkType.TABLE for chunk in parsed.chunks)
    assert any("alarm = pressure" in chunk.text for chunk in parsed.chunks)
    assert all(chunk.title_path for chunk in parsed.chunks)
    _assert_frozen_contract(parsed)


def test_plain_text_decodes_gb18030_without_lossy_replacement() -> None:
    text = "腔体清洁后检查首片。\n\n复核压力和 RF match。"

    parsed = build_dispatcher(ProcessPayloadStore()).parse(
        "checklist.txt",
        text.encode("gb18030"),
        declared_media_type="text/plain",
        correlation_id="txt-1",
    )

    assert parsed.detected_language == "zh"
    assert len(parsed.chunks) == 1
    assert "RF match" in parsed.chunks[0].text
    assert parsed.warnings[0].code == "INGEST_WARNING_TEXT_ENCODING_FALLBACK"


def test_html_removes_noise_keeps_structure_and_never_fetches_assets() -> None:
    html = b"""<!doctype html><html><head><title>FDC Alarm Guide</title>
    <script>secret()</script></head><body><nav>menu</nav><h1>Pressure Alarm</h1>
    <p>Check chamber pressure.</p><img src='https://example.com/remote.png'>
    <table><tr><th>Signal</th><th>Limit</th></tr><tr><td>Pressure</td><td>12</td></tr></table>
    </body></html>"""

    parsed = build_dispatcher(ProcessPayloadStore()).parse(
        "guide.html",
        html,
        declared_media_type="text/html",
        correlation_id="html-1",
    )

    assert parsed.detected_title == "FDC Alarm Guide"
    assert "secret" not in parsed.normalized_markdown
    assert "menu" not in parsed.normalized_markdown
    assert len(parsed.tables) == 1
    assert not parsed.images
    assert {warning.code for warning in parsed.warnings} == {
        "INGEST_WARNING_HTML_EXTERNAL_ASSETS_SKIPPED"
    }


def test_csv_detects_delimiter_headers_units_and_rows() -> None:
    content = b"Pressure(kPa);RF Match(%);Status\n10.2;98;OK\n12.1;82;ALARM\n"

    parsed = build_dispatcher(ProcessPayloadStore()).parse(
        "fdc.csv",
        content,
        declared_media_type="text/csv",
        correlation_id="csv-1",
    )

    assert parsed.source_format is SourceFormat.CSV
    assert parsed.tables[0].row_count == 2
    assert parsed.tables[0].headers[0] == "Pressure(kPa)"
    assert parsed.chunks[0].metadata["units"] == {"Pressure(kPa)": "kPa", "RF Match(%)": "%"}
    assert parsed.chunks[0].metadata["delimiter"] == ";"
