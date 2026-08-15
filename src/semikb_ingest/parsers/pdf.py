"""MinerU PDF adapter retaining content-list page and asset provenance."""

from __future__ import annotations

import time
from html import escape

from bs4 import BeautifulSoup

from semikb_ingest.assets import ProcessPayloadStore, inspect_image
from semikb_ingest.models import (
    CaptionSource,
    ImageAssetDraft,
    ParseWarning,
    SourceFormat,
    SourceLocation,
    TableAssetDraft,
)
from semikb_ingest.parsers.common import (
    complete_document,
    matrix_to_markdown,
    normalized_text,
)
from semikb_ingest.parsers.registry import ParseRequest
from semikb_ingest.parsers.textual import _markdown_blocks
from semikb_ingest.providers import PdfProvider
from semikb_ingest.structure import BlockKind, StructuredBlock


class PdfMinerUParser:
    parser_id = "pdf-mineru-v1"
    parser_version = "1.0.0"
    source_format = SourceFormat.PDF

    def __init__(
        self,
        payload_store: ProcessPayloadStore,
        provider: PdfProvider,
    ) -> None:
        if not isinstance(provider, PdfProvider):
            raise TypeError("The PDF parser requires a PdfProvider.")
        self.payload_store = payload_store
        self.provider = provider

    def parse(self, request: ParseRequest):
        started = time.monotonic()
        result = self.provider.parse_pdf(
            filename=request.filename,
            content=request.content,
            correlation_id=request.correlation_id,
        )
        blocks: list[StructuredBlock] = []
        images: list[ImageAssetDraft] = []
        tables: list[TableAssetDraft] = []
        warnings: list[ParseWarning] = []
        image_by_path: dict[str, ImageAssetDraft] = {}
        for provider_image in result.images:
            inspection = inspect_image(provider_image.content)
            payload = self.payload_store.put(
                provider_image.filename,
                provider_image.content_type,
                provider_image.content,
            )
            asset = ImageAssetDraft(
                asset_id=f"image_{len(images) + 1:04d}",
                payload=payload,
                image_type=(
                    f"pdf_page_asset:{inspection.width}x{inspection.height}"
                    if "page" in provider_image.path.lower()
                    else f"pdf_extracted:{inspection.width}x{inspection.height}"
                ),
                caption=provider_image.caption or provider_image.filename,
                caption_source=CaptionSource.MINERU,
                caption_confidence=0.85 if provider_image.caption else 0.5,
                location=SourceLocation(page_number=provider_image.page_number),
            )
            images.append(asset)
            image_by_path[provider_image.path] = asset

        heading_levels: list[str] = []
        detected_title: str | None = None
        if result.content_items:
            for item in result.content_items:
                location = SourceLocation(
                    section_path=tuple(value for value in heading_levels if value),
                    page_number=item.page_number,
                )
                if item.heading_level and item.text.strip():
                    level = item.heading_level
                    title = normalized_text(item.text)
                    heading_levels[level - 1 :] = []
                    while len(heading_levels) < level:
                        heading_levels.append("")
                    heading_levels[level - 1] = title
                    detected_title = detected_title or title
                    path = tuple(value for value in heading_levels if value)
                    blocks.append(
                        StructuredBlock(
                            block_id=f"block_{len(blocks) + 1:04d}",
                            kind=BlockKind.HEADING,
                            text=title,
                            heading_path=path,
                            location=SourceLocation(
                                section_path=path, page_number=item.page_number
                            ),
                            metadata={"level": level},
                        )
                    )
                    continue
                path = tuple(value for value in heading_levels if value)
                location = location.model_copy(update={"section_path": path})
                if item.kind == "table" and (item.table_html.strip() or item.text.strip()):
                    rows = self._html_rows(item.table_html)
                    markdown, headers = matrix_to_markdown(rows)
                    markdown = markdown or normalized_text(item.text) or item.table_caption
                    html = (
                        item.table_html.strip()
                        or f"<table><tr><td>{escape(markdown)}</td></tr></table>"
                    )
                    asset_id = f"table_{len(tables) + 1:04d}"
                    table = TableAssetDraft(
                        asset_id=asset_id,
                        title=item.table_caption or " / ".join(path) or f"Table {len(tables) + 1}",
                        html=html,
                        markdown=markdown,
                        headers=headers,
                        row_count=max(len(rows) - 1, 0),
                        column_count=max((len(row) for row in rows), default=1),
                        location=location,
                    )
                    tables.append(table)
                    blocks.append(
                        StructuredBlock(
                            block_id=f"block_{len(blocks) + 1:04d}",
                            kind=BlockKind.TABLE,
                            text=markdown,
                            heading_path=path,
                            location=location,
                            table_asset_ids=(asset_id,),
                        )
                    )
                elif item.image_path:
                    asset = image_by_path.get(item.image_path)
                    if asset:
                        image_text = item.text or asset.caption or asset.payload.filename
                        blocks.append(
                            StructuredBlock(
                                block_id=f"block_{len(blocks) + 1:04d}",
                                kind=BlockKind.IMAGE,
                                text=image_text,
                                heading_path=path,
                                location=location,
                                image_asset_ids=(asset.asset_id,),
                            )
                        )
                elif item.text.strip():
                    blocks.append(
                        StructuredBlock(
                            block_id=f"block_{len(blocks) + 1:04d}",
                            kind=BlockKind.PARAGRAPH,
                            text=normalized_text(item.text),
                            heading_path=path,
                            location=location,
                        )
                    )
        else:
            fallback_blocks, fallback_tables, detected_title = _markdown_blocks(result.markdown)
            blocks.extend(fallback_blocks)
            tables.extend(fallback_tables)
            for asset in images:
                blocks.append(
                    StructuredBlock(
                        block_id=f"block_{len(blocks) + 1:04d}",
                        kind=BlockKind.IMAGE,
                        text=asset.caption or asset.payload.filename,
                        location=asset.location,
                        image_asset_ids=(asset.asset_id,),
                    )
                )
            warnings.append(
                ParseWarning(
                    code="INGEST_WARNING_PDF_PAGE_MAP_UNAVAILABLE",
                    safe_message=(
                        "MinerU returned Markdown without a content-list page map; page-level "
                        "locations are unavailable for text chunks."
                    ),
                )
            )

        referenced = {asset_id for block in blocks for asset_id in block.image_asset_ids}
        for asset in images:
            if asset.asset_id not in referenced:
                blocks.append(
                    StructuredBlock(
                        block_id=f"block_{len(blocks) + 1:04d}",
                        kind=BlockKind.IMAGE,
                        text=asset.caption or asset.payload.filename,
                        location=asset.location,
                        image_asset_ids=(asset.asset_id,),
                    )
                )
        return complete_document(
            request=request,
            source_format=self.source_format,
            parser_name=self.parser_id,
            parser_version=self.parser_version,
            provider_name=self.provider.provider_name,
            provider_version=self.provider.provider_version,
            blocks=blocks,
            images=images,
            tables=tables,
            warnings=warnings,
            detected_title=detected_title or request.filename.rsplit(".", 1)[0],
            normalized_markdown=result.markdown,
            pages=result.pages,
            started_at=started,
        )

    @staticmethod
    def _html_rows(table_html: str) -> list[list[str]]:
        if not table_html.strip():
            return []
        soup = BeautifulSoup(table_html, "lxml")
        return [
            [normalized_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
            for row in soup.find_all("tr")
            if row.find_all(["th", "td"])
        ]
