from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from semikb_ingest import ChunkType, IngestError, IngestErrorCode, build_dispatcher
from semikb_ingest.assets import ProcessPayloadStore
from semikb_ingest.providers import (
    MinerUContentItem,
    MinerUImage,
    MinerUPdfClient,
    MinerUPdfConfig,
    MinerUPdfResult,
    ProviderRegistry,
    QwenVisionClient,
    QwenVisionConfig,
    VisionAnalysis,
)

_CONTRACT = json.loads(
    Path("docs/evidence/t9-4-4-1/semikb-ingest-contract-v1.schema.json").read_text(encoding="utf-8")
)


def _assert_frozen_contract(parsed) -> None:
    Draft202012Validator(_CONTRACT).validate(parsed.model_dump(mode="json"))


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (48, 48), (230, 40, 40)).save(output, format="PNG")
    return output.getvalue()


@dataclass
class FakeVisionProvider:
    provider_name: str = "qwen-vl"
    provider_version: str = "qwen3.7-plus"

    def analyze_image(self, **_kwargs) -> VisionAnalysis:
        return VisionAnalysis(
            caption="晶圆边缘出现红色环状缺陷分布。",
            ocr_text="ETCH-03",
            detection_summary="边缘缺陷集中，中心区域相对正常。",
            confidence=0.91,
            detected_language="zh",
        )


@dataclass
class FakePdfProvider:
    provider_name: str = "mineru"
    provider_version: str = "vlm"

    def parse_pdf(self, **_kwargs) -> MinerUPdfResult:
        image = _png_bytes()
        return MinerUPdfResult(
            markdown="# ETCH SOP\n\nPressure alarm.\n",
            pages=2,
            content_items=(
                MinerUContentItem(kind="text", text="ETCH SOP", heading_level=1, page_number=1),
                MinerUContentItem(kind="text", text="Pressure alarm.", page_number=1),
                MinerUContentItem(
                    kind="table",
                    table_html=(
                        "<table><tr><th>Signal</th><th>Limit</th></tr>"
                        "<tr><td>Pressure</td><td>12</td></tr></table>"
                    ),
                    table_caption="Alarm limits",
                    page_number=2,
                ),
                MinerUContentItem(kind="image", image_path="images/wafer.png", page_number=2),
            ),
            images=(
                MinerUImage(
                    path="images/wafer.png",
                    filename="wafer.png",
                    content_type="image/png",
                    content=image,
                    caption="Wafer edge map",
                    page_number=2,
                ),
            ),
        )


def _provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register(FakeVisionProvider())
    registry.register(FakePdfProvider())
    return registry


def test_image_adapter_returns_real_asset_and_image_text() -> None:
    payload_store = ProcessPayloadStore()
    parsed = build_dispatcher(payload_store, _provider_registry()).parse(
        "wafer.png",
        _png_bytes(),
        declared_media_type="image/png",
        correlation_id="image-1",
    )

    assert parsed.provenance.provider_version == "qwen3.7-plus"
    assert parsed.chunks[0].chunk_type is ChunkType.IMAGE_TEXT
    assert "ETCH-03" in parsed.chunks[0].text
    assert parsed.images[0].caption_confidence == 0.91
    assert payload_store.read(parsed.images[0].payload) == _png_bytes()
    _assert_frozen_contract(parsed)


def test_pdf_adapter_retains_page_table_and_image_references() -> None:
    payload_store = ProcessPayloadStore()
    parsed = build_dispatcher(payload_store, _provider_registry()).parse(
        "sop.pdf",
        b"%PDF-1.7\nsynthetic",
        declared_media_type="application/pdf",
        correlation_id="pdf-1",
    )

    assert parsed.metrics.pages == 2
    assert len(parsed.tables) == 1
    assert len(parsed.images) == 1
    assert {chunk.location.page_number for chunk in parsed.chunks} == {1, 2}
    assert parsed.tables[0].location.page_number == 2
    assert parsed.images[0].related_chunk_draft_ids
    _assert_frozen_contract(parsed)


def test_remote_formats_have_no_parser_when_provider_is_not_configured() -> None:
    dispatcher = build_dispatcher(ProcessPayloadStore())

    with pytest.raises(IngestError) as pdf_error:
        dispatcher.parse(
            "sop.pdf",
            b"%PDF-1.7",
            declared_media_type="application/pdf",
            correlation_id="missing-pdf",
        )
    assert pdf_error.value.code is IngestErrorCode.PARSER_NOT_CONFIGURED

    with pytest.raises(IngestError) as image_error:
        dispatcher.parse(
            "wafer.png",
            _png_bytes(),
            declared_media_type="image/png",
            correlation_id="missing-image",
        )
    assert image_error.value.code is IngestErrorCode.PARSER_NOT_CONFIGURED


def test_qwen_vision_client_sends_qwen37_plus_and_validates_json() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "caption": "Wafer map",
                                    "ocr_text": "ETCH-03",
                                    "detection_summary": "Edge ring",
                                    "confidence": 0.88,
                                    "detected_language": "en",
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = QwenVisionClient(
        QwenVisionConfig("https://example.test/v1", "secret", model="qwen3.7-plus"),
        transport=httpx.MockTransport(handler),
    )
    result = client.analyze_image(
        filename="wafer.png",
        content_type="image/png",
        content=_png_bytes(),
        correlation_id="qwen-1",
    )

    assert captured["model"] == "qwen3.7-plus"
    assert str(captured["messages"][1]["content"][0]["image_url"]["url"]).startswith(
        "data:image/png;base64,"
    )
    assert result.detection_summary == "Edge ring"


def test_qwen_vision_client_rejects_unstructured_provider_output() -> None:
    client = QwenVisionClient(
        QwenVisionConfig("https://example.test/v1", "secret"),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not-json"}}]},
            )
        ),
    )

    with pytest.raises(IngestError) as captured:
        client.analyze_image(
            filename="wafer.png",
            content_type="image/png",
            content=_png_bytes(),
            correlation_id="qwen-bad",
        )
    assert captured.value.code is IngestErrorCode.PARSE_FAILED


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [(429, IngestErrorCode.PARSER_UNAVAILABLE), (503, IngestErrorCode.PARSER_UNAVAILABLE)],
)
def test_qwen_vision_client_maps_transient_http_failures(
    status_code: int,
    expected_code: IngestErrorCode,
) -> None:
    client = QwenVisionClient(
        QwenVisionConfig("https://example.test/v1", "secret"),
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(status_code, json={"error": "busy"})
        ),
    )

    with pytest.raises(IngestError) as captured:
        client.analyze_image(
            filename="wafer.png",
            content_type="image/png",
            content=_png_bytes(),
            correlation_id="qwen-busy",
        )
    assert captured.value.code is expected_code


def test_qwen_vision_client_maps_timeout_without_leaking_request() -> None:
    def timeout(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider timeout")

    client = QwenVisionClient(
        QwenVisionConfig("https://example.test/v1", "secret"),
        transport=httpx.MockTransport(timeout),
    )

    with pytest.raises(IngestError) as captured:
        client.analyze_image(
            filename="wafer.png",
            content_type="image/png",
            content=_png_bytes(),
            correlation_id="qwen-timeout",
        )
    assert captured.value.code is IngestErrorCode.PARSER_TIMEOUT
    assert "secret" not in captured.value.safe_message


def test_qwen_vision_retries_transient_failure_then_records_success() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "caption": "Wafer map",
                                    "ocr_text": "",
                                    "detection_summary": "Edge ring",
                                    "confidence": 0.8,
                                }
                            )
                        }
                    }
                ]
            },
        )

    client = QwenVisionClient(
        QwenVisionConfig(
            "https://example.test/v1",
            "secret",
            backoff_base_seconds=0,
            backoff_max_seconds=0,
        ),
        transport=httpx.MockTransport(handler),
    )

    result = client.analyze_image(
        filename="wafer.png",
        content_type="image/png",
        content=_png_bytes(),
        correlation_id="qwen-retry",
    )

    assert result.caption == "Wafer map"
    assert calls == 2
    assert [attempt.outcome for attempt in client.last_attempts] == ["retrying", "succeeded"]


def test_mineru_archive_reader_retains_content_list_pages_and_assets() -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("result/full.md", "# SOP\n\n![map](images/map.png)\n")
        archive.writestr(
            "result/content_list.json",
            json.dumps(
                [
                    {"type": "text", "text": "SOP", "text_level": 1, "page_idx": 0},
                    {
                        "type": "image",
                        "img_path": "images/map.png",
                        "image_caption": ["Wafer map"],
                        "page_idx": 1,
                    },
                ]
            ),
        )
        archive.writestr("result/images/map.png", _png_bytes())

    result = MinerUPdfClient.read_archive(archive_bytes.getvalue())

    assert result.pages == 2
    assert result.content_items[0].heading_level == 1
    assert result.images[0].page_number == 2
    assert result.images[0].caption == "Wafer map"


def test_mineru_archive_reader_rejects_path_traversal() -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("result/full.md", "# SOP")
        archive.writestr("../outside.png", _png_bytes())

    with pytest.raises(IngestError) as captured:
        MinerUPdfClient.read_archive(archive_bytes.getvalue())
    assert captured.value.code is IngestErrorCode.ASSET_EXTRACTION_FAILED


def test_mineru_retries_idempotent_transfer_but_not_batch_creation() -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("result/full.md", "# SOP\n\nPressure alarm.")

    calls: list[tuple[str, str]] = []
    upload_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal upload_calls
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "file_urls": ["https://upload.test/source"],
                        "batch_id": "batch-1",
                    },
                },
            )
        if request.url.host == "upload.test":
            upload_calls += 1
            return httpx.Response(503 if upload_calls == 1 else 200)
        if "extract-results" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {"state": "done", "full_zip_url": "https://archive.test/result.zip"}
                        ]
                    },
                },
            )
        if request.url.host == "archive.test":
            return httpx.Response(200, content=archive_bytes.getvalue())
        raise AssertionError(request.url)

    client = MinerUPdfClient(
        MinerUPdfConfig(
            "https://mineru.test",
            "secret",
            poll_seconds=0,
            backoff_base_seconds=0,
            backoff_max_seconds=0,
        ),
        transport=httpx.MockTransport(handler),
    )

    result = client.parse_pdf(filename="sop.pdf", content=b"%PDF-1.7", correlation_id="pdf-1")

    assert "Pressure alarm" in result.markdown
    assert len([item for item in calls if item[0] == "POST"]) == 1
    assert upload_calls == 2
    assert [attempt.operation for attempt in client.last_attempts] == [
        "create_batch",
        "upload_source",
        "upload_source",
        "poll_batch",
        "download_archive",
    ]


def test_mineru_does_not_replay_ambiguous_batch_creation_failure() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": "busy"})

    client = MinerUPdfClient(
        MinerUPdfConfig(
            "https://mineru.test",
            "secret",
            max_attempts=3,
            backoff_base_seconds=0,
            backoff_max_seconds=0,
        ),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(IngestError) as captured:
        client.parse_pdf(filename="sop.pdf", content=b"%PDF-1.7", correlation_id="pdf-2")

    assert calls == 1
    assert captured.value.code is IngestErrorCode.PARSER_UNAVAILABLE
    assert len(captured.value.provider_attempts) == 1
