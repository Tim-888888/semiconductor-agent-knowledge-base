"""Public models emitted by parsers without governance or storage concerns."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

CONTRACT_VERSION = "semikb-ingest-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    PPTX = "pptx"
    IMAGE = "image"
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"


class ChunkType(StrEnum):
    TEXT = "text"
    TABLE = "table"
    IMAGE_TEXT = "image_text"


class CaptionSource(StrEnum):
    HUMAN = "human"
    PARSER = "parser"
    OCR = "ocr"
    VLM = "vlm"
    MINERU = "mineru"


class SourceLocation(StrictModel):
    section_path: tuple[str, ...] = ()
    page_number: Annotated[int, Field(ge=1)] | None = None
    slide_number: Annotated[int, Field(ge=1)] | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    row_start: Annotated[int, Field(ge=1)] | None = None
    row_end: Annotated[int, Field(ge=1)] | None = None
    bbox: tuple[float, float, float, float] | None = None


class BinaryPayload(StrictModel):
    handle: Annotated[str, Field(min_length=1)]
    filename: Annotated[str, Field(min_length=1)]
    content_type: Annotated[str, Field(min_length=1)]
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ParseProvenance(StrictModel):
    parser_name: Annotated[str, Field(min_length=1)]
    parser_version: Annotated[str, Field(min_length=1)]
    provider_name: str | None
    provider_version: str | None
    upstream_project: str | None
    upstream_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] | None
    source_filename: Annotated[str, Field(min_length=1)]
    source_media_type: Annotated[str, Field(min_length=1)]
    source_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    detected_format: Annotated[str, Field(min_length=1)]


class ChunkDraft(StrictModel):
    draft_id: Annotated[str, Field(pattern=r"^draft_[A-Za-z0-9._:-]+$")]
    chunk_type: ChunkType
    text: Annotated[str, Field(min_length=1)]
    title_path: tuple[str, ...] = ()
    location: SourceLocation = Field(default_factory=SourceLocation)
    parent_draft_id: str | None = None
    image_asset_ids: tuple[str, ...] = ()
    table_asset_ids: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("image_asset_ids", "table_asset_ids")
    @classmethod
    def validate_unique_asset_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Asset references must be unique within one chunk.")
        return value


class ImageAssetDraft(StrictModel):
    asset_id: Annotated[str, Field(min_length=1)]
    payload: BinaryPayload
    image_type: Annotated[str, Field(min_length=1)]
    caption: str
    caption_source: CaptionSource
    caption_confidence: Annotated[float, Field(ge=0, le=1)]
    ocr_text: str = ""
    detection_summary: str = ""
    location: SourceLocation = Field(default_factory=SourceLocation)
    related_chunk_draft_ids: tuple[str, ...] = ()

    @field_validator("related_chunk_draft_ids")
    @classmethod
    def validate_unique_chunk_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Related chunk references must be unique.")
        return value


class TableAssetDraft(StrictModel):
    asset_id: Annotated[str, Field(min_length=1)]
    title: str
    html: Annotated[str, Field(min_length=1)]
    markdown: Annotated[str, Field(min_length=1)]
    headers: tuple[str, ...] = ()
    row_count: Annotated[int, Field(ge=0)]
    column_count: Annotated[int, Field(ge=0)]
    location: SourceLocation = Field(default_factory=SourceLocation)
    related_chunk_draft_ids: tuple[str, ...] = ()

    @field_validator("related_chunk_draft_ids")
    @classmethod
    def validate_unique_chunk_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("Related chunk references must be unique.")
        return value


class ParseWarning(StrictModel):
    code: Annotated[str, Field(pattern=r"^INGEST_WARNING_[A-Z0-9_]+$")]
    safe_message: Annotated[str, Field(min_length=1)]
    location: SourceLocation = Field(default_factory=SourceLocation)


class ParseMetrics(StrictModel):
    pages: Annotated[int, Field(ge=0)] = 0
    slides: Annotated[int, Field(ge=0)] = 0
    sheets: Annotated[int, Field(ge=0)] = 0
    chunks: Annotated[int, Field(ge=0)] = 0
    images: Annotated[int, Field(ge=0)] = 0
    tables: Annotated[int, Field(ge=0)] = 0
    duration_ms: Annotated[float, Field(ge=0)] = 0


class ParsedDocument(StrictModel):
    contract_version: Literal["semikb-ingest-v1"] = CONTRACT_VERSION
    source_format: SourceFormat
    normalized_markdown: str
    detected_title: str | None = None
    detected_language: str | None = None
    chunks: tuple[ChunkDraft, ...] = ()
    images: tuple[ImageAssetDraft, ...] = ()
    tables: tuple[TableAssetDraft, ...] = ()
    provenance: ParseProvenance
    warnings: tuple[ParseWarning, ...] = ()
    metrics: ParseMetrics = Field(default_factory=ParseMetrics)

    @model_validator(mode="after")
    def validate_references(self) -> ParsedDocument:
        chunk_ids = {chunk.draft_id for chunk in self.chunks}
        image_ids = {image.asset_id for image in self.images}
        table_ids = {table.asset_id for table in self.tables}
        if len(chunk_ids) != len(self.chunks):
            raise ValueError("Chunk draft identifiers must be unique.")
        if len(image_ids) != len(self.images):
            raise ValueError("Image asset identifiers must be unique.")
        if len(table_ids) != len(self.tables):
            raise ValueError("Table asset identifiers must be unique.")

        for chunk in self.chunks:
            if chunk.parent_draft_id is not None and chunk.parent_draft_id not in chunk_ids:
                raise ValueError("A chunk parent reference does not exist in this document.")
            if not set(chunk.image_asset_ids).issubset(image_ids):
                raise ValueError("A chunk references an image asset that does not exist.")
            if not set(chunk.table_asset_ids).issubset(table_ids):
                raise ValueError("A chunk references a table asset that does not exist.")
        for image in self.images:
            if not set(image.related_chunk_draft_ids).issubset(chunk_ids):
                raise ValueError("An image references a chunk draft that does not exist.")
        for table in self.tables:
            if not set(table.related_chunk_draft_ids).issubset(chunk_ids):
                raise ValueError("A table references a chunk draft that does not exist.")
        return self
