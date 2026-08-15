"""Typed provider results consumed by PDF and image adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import Field

from semikb_ingest.models import StrictModel


class VisionAnalysis(StrictModel):
    caption: str = Field(min_length=1)
    ocr_text: str = ""
    detection_summary: str = ""
    confidence: float = Field(default=0.8, ge=0, le=1)
    detected_language: str | None = None


@runtime_checkable
class VisionProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze_image(
        self,
        *,
        filename: str,
        content_type: str,
        content: bytes,
        correlation_id: str,
    ) -> VisionAnalysis: ...


class MinerUImage(StrictModel):
    path: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    content_type: str = Field(min_length=1)
    content: bytes = Field(min_length=1)
    caption: str = ""
    page_number: int | None = Field(default=None, ge=1)


class MinerUContentItem(StrictModel):
    kind: str = Field(min_length=1)
    text: str = ""
    heading_level: int | None = Field(default=None, ge=1, le=6)
    page_number: int | None = Field(default=None, ge=1)
    image_path: str | None = None
    table_html: str = ""
    table_caption: str = ""


class MinerUPdfResult(StrictModel):
    markdown: str
    content_items: tuple[MinerUContentItem, ...] = ()
    images: tuple[MinerUImage, ...] = ()
    pages: int = Field(default=0, ge=0)


@runtime_checkable
class PdfProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def parse_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        correlation_id: str,
    ) -> MinerUPdfResult: ...
