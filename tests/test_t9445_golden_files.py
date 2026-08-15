from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from semikb.config import Settings
from semikb.rag_ingestion.semikb_adapter import SemikbIngestAdapter
from semikb_ingest.providers import (
    MinerUContentItem,
    MinerUImage,
    MinerUPdfResult,
    ProviderRegistry,
    VisionAnalysis,
)

GOLDEN_ROOT = Path("data/t9445_golden")


@dataclass
class GoldenVisionProvider:
    provider_name: str = "qwen-vl"
    provider_version: str = "qwen3.7-plus"

    def analyze_image(self, **_kwargs) -> VisionAnalysis:
        return VisionAnalysis(
            caption="ETCH-03 Chamber B wafer edge ring map.",
            ocr_text="T9445 IMAGE EDGE RING 57",
            detection_summary="Edge-ring defect concentration; center region stable.",
            confidence=0.94,
            detected_language="en",
        )


@dataclass
class GoldenPdfProvider:
    image: bytes
    provider_name: str = "mineru"
    provider_version: str = "vlm"

    def parse_pdf(self, **_kwargs) -> MinerUPdfResult:
        return MinerUPdfResult(
            markdown=(
                "# T9445 PDF Controlled Case\n\n"
                "Control token T9445-PDF-ESCALATE-42.\n\n"
                "## Release Gate\n\nRelease after leak check.\n"
            ),
            pages=2,
            content_items=(
                MinerUContentItem(
                    kind="text",
                    text="T9445 PDF Controlled Case",
                    heading_level=1,
                    page_number=1,
                ),
                MinerUContentItem(
                    kind="text",
                    text="Control token T9445-PDF-ESCALATE-42.",
                    page_number=1,
                ),
                MinerUContentItem(
                    kind="text",
                    text="Release Gate",
                    heading_level=2,
                    page_number=2,
                ),
                MinerUContentItem(
                    kind="table",
                    table_html=(
                        "<table><tr><th>Signal</th><th>Limit</th></tr>"
                        "<tr><td>Pressure</td><td>12 Pa</td></tr></table>"
                    ),
                    table_caption="Release limits",
                    page_number=2,
                ),
                MinerUContentItem(
                    kind="text",
                    text="Release after leak check.",
                    page_number=2,
                ),
                MinerUContentItem(
                    kind="image",
                    text="Embedded wafer edge-ring evidence",
                    image_path="images/wafer.png",
                    page_number=2,
                ),
            ),
            images=(
                MinerUImage(
                    path="images/wafer.png",
                    filename="wafer.png",
                    content_type="image/png",
                    content=self.image,
                    caption="Embedded wafer edge-ring evidence",
                    page_number=2,
                ),
            ),
        )


def _manifest() -> dict[str, object]:
    return json.loads((GOLDEN_ROOT / "manifest.json").read_text(encoding="utf-8"))


def _corpus(parsed) -> str:
    values = [parsed.normalized_markdown]
    for chunk in parsed.chunks:
        values.extend((chunk.text, *chunk.title_path))
        values.extend(value for value in chunk.location.model_dump().values() if isinstance(value, str))
    for table in parsed.tables:
        values.extend((table.title, table.markdown, *table.headers))
    for image in parsed.images:
        values.extend((image.caption, image.ocr_text, image.detection_summary))
    return "\n".join(values).lower()


def test_committed_golden_files_match_the_frozen_manifest() -> None:
    manifest = _manifest()
    entries = manifest["entries"]
    assert manifest["golden_version"] == "t9-4.4.5-golden-v1"
    assert manifest["license"] == "CC0-1.0"
    assert len(entries) == 9
    assert {entry["category"] for entry in entries} == {
        "markdown",
        "text",
        "html",
        "pdf",
        "docx",
        "xlsx",
        "csv",
        "pptx",
        "image",
    }
    for entry in entries:
        content = (GOLDEN_ROOT / entry["filename"]).read_bytes()
        assert len(content) == entry["size_bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]


def test_every_golden_file_satisfies_the_unified_parse_contract() -> None:
    image = (GOLDEN_ROOT / "09_edge_ring_wafer.png").read_bytes()
    providers = ProviderRegistry()
    providers.register(GoldenPdfProvider(image))
    providers.register(GoldenVisionProvider())
    adapter = SemikbIngestAdapter(Settings(_env_file=None), providers)

    for entry in _manifest()["entries"]:
        path = GOLDEN_ROOT / entry["filename"]
        session = adapter.parse(
            filename=entry["filename"],
            content=path.read_bytes(),
            declared_media_type=entry["content_type"],
            correlation_id=f"golden-{entry['category']}",
        )
        try:
            parsed = session.document
            assert parsed.provenance.parser_name == entry["parser_name"]
            assert len(parsed.chunks) >= entry["min_chunks"]
            assert len(parsed.images) >= entry["min_images"]
            assert len(parsed.tables) >= entry["min_tables"]
            corpus = _corpus(parsed)
            assert all(term.lower() in corpus for term in entry["required_terms"])
            assert all(term.lower() not in corpus for term in entry.get("forbidden_terms", []))
            if entry["category"] == "pdf":
                assert parsed.metrics.pages == 2
                assert {chunk.location.page_number for chunk in parsed.chunks} == {1, 2}
            if entry["category"] == "xlsx":
                assert parsed.metrics.sheets == 2
                assert {table.location.sheet_name for table in parsed.tables} == {
                    "FDC_Limits",
                    "Recipe_Audit",
                }
            if entry["category"] == "pptx":
                assert parsed.metrics.slides == 2
                assert {chunk.location.slide_number for chunk in parsed.chunks} == {1, 2}
        finally:
            session.discard_remaining()


def test_workbench_file_picker_exposes_every_supported_extension() -> None:
    source = Path("web/src/components/IngestionPanel.tsx").read_text(encoding="utf-8")
    for extension in (
        ".pdf",
        ".docx",
        ".xlsx",
        ".csv",
        ".pptx",
        ".png",
        ".jpg",
        ".jpeg",
        ".md",
        ".txt",
        ".html",
        ".htm",
    ):
        assert extension in source
