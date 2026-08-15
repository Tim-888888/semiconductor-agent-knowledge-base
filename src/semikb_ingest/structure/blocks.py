"""Internal blocks shared by format adapters and chunking strategies."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, JsonValue

from semikb_ingest.models import SourceLocation, StrictModel


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"


class StructuredBlock(StrictModel):
    block_id: str = Field(min_length=1)
    kind: BlockKind
    text: str
    heading_path: tuple[str, ...] = ()
    location: SourceLocation = Field(default_factory=SourceLocation)
    image_asset_ids: tuple[str, ...] = ()
    table_asset_ids: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
