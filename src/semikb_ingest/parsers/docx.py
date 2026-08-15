"""DOCX block-order adapter for headings, prose, lists, tables, and images."""

from __future__ import annotations

import hashlib
import io
import posixpath
import re
import time
from dataclasses import dataclass

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.ns import qn
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from semikb_ingest.assets import ProcessPayloadStore, inspect_image
from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.models import (
    CaptionSource,
    ImageAssetDraft,
    SourceFormat,
    SourceLocation,
    TableAssetDraft,
)
from semikb_ingest.parsers.common import (
    complete_document,
    matrix_to_html,
    matrix_to_markdown,
    normalized_text,
)
from semikb_ingest.parsers.registry import ParseRequest
from semikb_ingest.structure import BlockKind, StructuredBlock


@dataclass(frozen=True, slots=True)
class DocxLimits:
    max_blocks: int = 100_000
    max_tables: int = 5_000
    max_images: int = 5_000


class DocxStructuredParser:
    parser_id = "docx-structured-v1"
    parser_version = "1.0.0"
    source_format = SourceFormat.DOCX

    def __init__(
        self,
        payload_store: ProcessPayloadStore,
        limits: DocxLimits | None = None,
    ) -> None:
        self.payload_store = payload_store
        self.limits = limits or DocxLimits()

    def parse(self, request: ParseRequest):
        started = time.monotonic()
        try:
            document = Document(io.BytesIO(request.content))
        except Exception as exc:
            raise IngestError(
                IngestErrorCode.CORRUPT_DOCUMENT,
                "The DOCX document could not be opened.",
            ) from exc

        blocks: list[StructuredBlock] = []
        images: list[ImageAssetDraft] = []
        tables: list[TableAssetDraft] = []
        image_by_hash: dict[str, ImageAssetDraft] = {}
        heading_levels: list[str] = []

        for item in self._iter_blocks(document):
            if len(blocks) > self.limits.max_blocks:
                self._raise_limit()
            if isinstance(item, Paragraph):
                self._append_paragraph(
                    document,
                    item,
                    blocks,
                    images,
                    image_by_hash,
                    heading_levels,
                )
            else:
                if len(tables) >= self.limits.max_tables:
                    self._raise_limit()
                self._append_table(
                    document,
                    item,
                    blocks,
                    images,
                    tables,
                    image_by_hash,
                    tuple(value for value in heading_levels if value),
                )
        title = normalized_text(document.core_properties.title) or (
            next((block.text for block in blocks if block.kind is BlockKind.HEADING), None)
        )
        return complete_document(
            request=request,
            source_format=self.source_format,
            parser_name=self.parser_id,
            parser_version=self.parser_version,
            provider_name=request.route.provider,
            provider_version="python-docx-1.2",
            blocks=blocks,
            images=images,
            tables=tables,
            detected_title=title,
            started_at=started,
            reference_knowhere=True,
        )

    @staticmethod
    def _iter_blocks(document: DocumentObject):
        for child in document.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, document)
            elif isinstance(child, CT_Tbl):
                yield Table(child, document)

    def _append_paragraph(
        self,
        document: DocumentObject,
        paragraph: Paragraph,
        blocks: list[StructuredBlock],
        images: list[ImageAssetDraft],
        image_by_hash: dict[str, ImageAssetDraft],
        heading_levels: list[str],
    ) -> None:
        level = self._heading_level(paragraph)
        paragraph_text = normalized_text(paragraph.text)
        if level and paragraph_text:
            heading_levels[level - 1 :] = []
            while len(heading_levels) < level:
                heading_levels.append("")
            heading_levels[level - 1] = paragraph_text
            path = tuple(value for value in heading_levels if value)
            blocks.append(
                StructuredBlock(
                    block_id=f"block_{len(blocks) + 1:04d}",
                    kind=BlockKind.HEADING,
                    text=paragraph_text,
                    heading_path=path,
                    location=SourceLocation(section_path=path),
                    metadata={"level": level},
                )
            )
        path = tuple(value for value in heading_levels if value)
        buffered_text: list[str] = []
        paragraph_kind = BlockKind.LIST_ITEM if self._is_list(paragraph) else BlockKind.PARAGRAPH

        def flush_text() -> None:
            value = normalized_text("".join(buffered_text))
            buffered_text.clear()
            if value and not level:
                blocks.append(
                    StructuredBlock(
                        block_id=f"block_{len(blocks) + 1:04d}",
                        kind=paragraph_kind,
                        text=value,
                        heading_path=path,
                        location=SourceLocation(section_path=path),
                    )
                )

        for run in paragraph.runs:
            buffered_text.append(run.text)
            run_images = self._run_images(document, run, path, images, image_by_hash)
            if run_images:
                flush_text()
            for asset in run_images:
                blocks.append(self._image_block(asset, blocks, path))
        flush_text()

    def _append_table(
        self,
        document: DocumentObject,
        table: Table,
        blocks: list[StructuredBlock],
        images: list[ImageAssetDraft],
        tables: list[TableAssetDraft],
        image_by_hash: dict[str, ImageAssetDraft],
        heading_path: tuple[str, ...],
    ) -> None:
        rows = [[normalized_text(cell.text) for cell in row.cells] for row in table.rows]
        markdown, headers = matrix_to_markdown(rows)
        if not markdown:
            return
        location = SourceLocation(section_path=heading_path)
        asset_id = f"table_{len(tables) + 1:04d}"
        table_asset = TableAssetDraft(
            asset_id=asset_id,
            title=" / ".join(heading_path) or f"Table {len(tables) + 1}",
            html=matrix_to_html(rows),
            markdown=markdown,
            headers=headers,
            row_count=max(len(rows) - 1, 0),
            column_count=max((len(row) for row in rows), default=0),
            location=location,
        )
        tables.append(table_asset)
        cell_assets: list[ImageAssetDraft] = []
        seen_cells: set[int] = set()
        for row in table.rows:
            for cell in row.cells:
                if id(cell._tc) in seen_cells:
                    continue
                seen_cells.add(id(cell._tc))
                cell_assets.extend(
                    self._cell_images(document, cell, heading_path, images, image_by_hash)
                )
        blocks.append(
            StructuredBlock(
                block_id=f"block_{len(blocks) + 1:04d}",
                kind=BlockKind.TABLE,
                text=markdown,
                heading_path=heading_path,
                location=location,
                image_asset_ids=tuple(dict.fromkeys(asset.asset_id for asset in cell_assets)),
                table_asset_ids=(asset_id,),
            )
        )
        for asset in cell_assets:
            blocks.append(self._image_block(asset, blocks, heading_path))

    def _cell_images(
        self,
        document: DocumentObject,
        cell: _Cell,
        heading_path: tuple[str, ...],
        images: list[ImageAssetDraft],
        image_by_hash: dict[str, ImageAssetDraft],
    ) -> list[ImageAssetDraft]:
        found: list[ImageAssetDraft] = []
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                found.extend(self._run_images(document, run, heading_path, images, image_by_hash))
        return found

    def _run_images(
        self,
        document: DocumentObject,
        run,
        heading_path: tuple[str, ...],
        images: list[ImageAssetDraft],
        image_by_hash: dict[str, ImageAssetDraft],
    ) -> list[ImageAssetDraft]:
        found: list[ImageAssetDraft] = []
        for blip in run._element.xpath(".//a:blip"):
            relationship_id = blip.get(qn("r:embed"))
            if not relationship_id or relationship_id not in document.part.related_parts:
                continue
            part = document.part.related_parts[relationship_id]
            content = bytes(part.blob)
            digest = hashlib.sha256(content).hexdigest()
            if digest in image_by_hash:
                found.append(image_by_hash[digest])
                continue
            if len(images) >= self.limits.max_images:
                self._raise_limit()
            inspection = inspect_image(content)
            original_name = posixpath.basename(str(part.partname))
            filename = original_name or f"image-{len(images) + 1}.{inspection.format}"
            payload = self.payload_store.put(filename, inspection.content_type, content)
            doc_properties = run._element.xpath(".//wp:docPr")
            alt_text = ""
            if doc_properties:
                alt_text = normalized_text(
                    doc_properties[0].get("descr") or doc_properties[0].get("title")
                )
            caption = alt_text or " / ".join(heading_path) or filename
            asset = ImageAssetDraft(
                asset_id=f"image_{len(images) + 1:04d}",
                payload=payload,
                image_type=f"docx_embedded:{inspection.width}x{inspection.height}",
                caption=caption,
                caption_source=CaptionSource.PARSER,
                caption_confidence=0.9 if alt_text else 0.55,
                location=SourceLocation(section_path=heading_path),
            )
            images.append(asset)
            image_by_hash[digest] = asset
            found.append(asset)
        return found

    @staticmethod
    def _image_block(
        asset: ImageAssetDraft,
        blocks: list[StructuredBlock],
        heading_path: tuple[str, ...],
    ) -> StructuredBlock:
        return StructuredBlock(
            block_id=f"block_{len(blocks) + 1:04d}",
            kind=BlockKind.IMAGE,
            text=asset.caption or asset.payload.filename,
            heading_path=heading_path,
            location=asset.location,
            image_asset_ids=(asset.asset_id,),
        )

    @staticmethod
    def _heading_level(paragraph: Paragraph) -> int | None:
        style = paragraph.style
        style_name = style.name if style is not None else ""
        style_id = style.style_id if style is not None else ""
        match = re.search(r"(?:Heading|标题)\s*([1-6])", f"{style_name} {style_id}", re.I)
        return int(match.group(1)) if match else None

    @staticmethod
    def _is_list(paragraph: Paragraph) -> bool:
        style_name = paragraph.style.name if paragraph.style is not None else ""
        properties = paragraph._p.pPr
        return "list" in style_name.lower() or (
            properties is not None and properties.numPr is not None
        )

    @staticmethod
    def _raise_limit() -> None:
        raise IngestError(
            IngestErrorCode.DOCUMENT_LIMIT_EXCEEDED,
            "The DOCX exceeds configured block, table, or image limits.",
        )
