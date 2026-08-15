"""Deterministic structure-aware chunks shared by all format adapters."""

from __future__ import annotations

from collections.abc import Sequence

from semikb_ingest.models import ChunkDraft, ChunkType, SourceLocation
from semikb_ingest.structure import BlockKind, StructuredBlock


class StructuredBlockChunker:
    """Keep table/image assets atomic while grouping adjacent prose by section."""

    chunker_version = "structured-blocks-v1"

    def __init__(self, max_chars: int = 2400) -> None:
        if max_chars < 256:
            raise ValueError("max_chars must be at least 256.")
        self.max_chars = max_chars

    def chunk(self, blocks: Sequence[StructuredBlock]) -> tuple[ChunkDraft, ...]:
        chunks: list[ChunkDraft] = []
        prose: list[StructuredBlock] = []

        def flush_prose() -> None:
            nonlocal prose
            if not prose:
                return
            text = "\n\n".join(block.text.strip() for block in prose if block.text.strip())
            if text:
                chunks.append(self._build_chunk(chunks, ChunkType.TEXT, text, prose))
            prose = []

        for block in blocks:
            if block.kind is BlockKind.HEADING:
                flush_prose()
                continue
            if block.kind in {BlockKind.TABLE, BlockKind.IMAGE}:
                flush_prose()
                chunk_type = (
                    ChunkType.TABLE if block.kind is BlockKind.TABLE else ChunkType.IMAGE_TEXT
                )
                if block.text.strip():
                    chunks.append(self._build_chunk(chunks, chunk_type, block.text, [block]))
                continue

            next_size = sum(len(item.text) for item in prose) + len(block.text)
            heading_changed = prose and prose[-1].heading_path != block.heading_path
            if prose and (heading_changed or next_size > self.max_chars):
                flush_prose()
            prose.append(block)

        flush_prose()
        return tuple(chunks)

    @staticmethod
    def _build_chunk(
        existing: Sequence[ChunkDraft],
        chunk_type: ChunkType,
        text: str,
        blocks: Sequence[StructuredBlock],
    ) -> ChunkDraft:
        first = blocks[0]
        image_ids = tuple(dict.fromkeys(item for block in blocks for item in block.image_asset_ids))
        table_ids = tuple(dict.fromkeys(item for block in blocks for item in block.table_asset_ids))
        block_ids = [block.block_id for block in blocks]
        block_kinds = [block.kind.value for block in blocks]
        metadata = dict(first.metadata)
        metadata.update({"block_ids": block_ids, "block_kinds": block_kinds})
        if len(blocks) > 1:
            block_metadata = [block.metadata for block in blocks if block.metadata]
            if block_metadata:
                metadata["block_metadata"] = block_metadata
        location = first.location
        if len(blocks) > 1:
            location = SourceLocation(
                section_path=first.location.section_path,
                page_number=first.location.page_number,
                slide_number=first.location.slide_number,
                sheet_name=first.location.sheet_name,
                cell_range=first.location.cell_range,
                row_start=first.location.row_start,
                row_end=blocks[-1].location.row_end or first.location.row_end,
                bbox=first.location.bbox,
            )
        return ChunkDraft(
            draft_id=f"draft_{len(existing) + 1:04d}",
            chunk_type=chunk_type,
            text=text.strip(),
            title_path=first.heading_path,
            location=location,
            image_asset_ids=image_ids,
            table_asset_ids=table_ids,
            metadata=metadata,
        )
