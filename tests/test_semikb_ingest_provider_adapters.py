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
