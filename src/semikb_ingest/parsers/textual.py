"""Structured Markdown, plain-text, and HTML adapters."""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify as markdownify_html

from semikb_ingest.models import (
    ParseWarning,
    SourceFormat,
    SourceLocation,
    TableAssetDraft,
)
from semikb_ingest.parsers.common import (
    complete_document,
    decode_text,
    matrix_to_html,
    matrix_to_markdown,
    normalized_text,
)
from semikb_ingest.parsers.registry import ParseRequest
from semikb_ingest.structure import BlockKind, StructuredBlock

_TABLE_DELIMITER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+(.+)$")


def _language(text: str) -> str | None:
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return None
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in visible)
    return "zh" if cjk / len(visible) >= 0.15 else "en"


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", stripped)]


def _markdown_blocks(
    text: str,
) -> tuple[list[StructuredBlock], list[TableAssetDraft], str | None]:
    lines = text.splitlines()
    blocks: list[StructuredBlock] = []
    tables: list[TableAssetDraft] = []
    heading_levels: list[str] = []
    detected_title: str | None = None
    index = 0

    def heading_path() -> tuple[str, ...]:
        return tuple(value for value in heading_levels if value)

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            index += 1
            continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            language = stripped[3:].strip()
            code: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(fence):
                code.append(lines[index])
                index += 1
            index += 1 if index < len(lines) else 0
            blocks.append(
                StructuredBlock(
                    block_id=f"block_{len(blocks) + 1:04d}",
                    kind=BlockKind.CODE,
                    text="\n".join(code).strip(),
                    heading_path=heading_path(),
                    location=SourceLocation(section_path=heading_path()),
                    metadata={"language": language},
                )
            )
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            title = normalized_text(heading_match.group(2))
            heading_levels[level - 1 :] = []
            while len(heading_levels) < level:
                heading_levels.append("")
            heading_levels[level - 1] = title
            if detected_title is None:
                detected_title = title
            blocks.append(
                StructuredBlock(
                    block_id=f"block_{len(blocks) + 1:04d}",
                    kind=BlockKind.HEADING,
                    text=title,
                    heading_path=heading_path(),
                    location=SourceLocation(section_path=heading_path()),
                    metadata={"level": level},
                )
            )
            index += 1
            continue

        if index + 1 < len(lines) and "|" in raw and _TABLE_DELIMITER.match(lines[index + 1]):
            table_lines = [raw]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                table_lines.append(lines[index])
                index += 1
            rows = [_markdown_cells(line) for line in table_lines]
            markdown, headers = matrix_to_markdown(rows)
            asset_id = f"table_{len(tables) + 1:04d}"
            location = SourceLocation(section_path=heading_path())
            tables.append(
                TableAssetDraft(
                    asset_id=asset_id,
                    title=" / ".join(heading_path()) or f"Table {len(tables) + 1}",
                    html=matrix_to_html(rows),
                    markdown=markdown,
                    headers=headers,
                    row_count=max(len(rows) - 1, 0),
                    column_count=max((len(row) for row in rows), default=0),
                    location=location,
                )
            )
            blocks.append(
                StructuredBlock(
                    block_id=f"block_{len(blocks) + 1:04d}",
                    kind=BlockKind.TABLE,
                    text=markdown,
                    heading_path=heading_path(),
                    location=location,
                    table_asset_ids=(asset_id,),
                )
            )
            continue

        list_match = _LIST_ITEM.match(raw)
        if list_match:
            blocks.append(
                StructuredBlock(
                    block_id=f"block_{len(blocks) + 1:04d}",
                    kind=BlockKind.LIST_ITEM,
                    text=normalized_text(list_match.group(1)),
                    heading_path=heading_path(),
                    location=SourceLocation(section_path=heading_path()),
                )
            )
            index += 1
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if candidate.startswith(("```", "~~~", "#")) or _LIST_ITEM.match(lines[index]):
                break
            if (
                index + 1 < len(lines)
                and "|" in lines[index]
                and _TABLE_DELIMITER.match(lines[index + 1])
            ):
                break
            paragraph.append(candidate)
            index += 1
        blocks.append(
            StructuredBlock(
                block_id=f"block_{len(blocks) + 1:04d}",
                kind=BlockKind.PARAGRAPH,
                text=normalized_text(" ".join(paragraph)),
                heading_path=heading_path(),
                location=SourceLocation(section_path=heading_path()),
            )
        )
    return blocks, tables, detected_title


class MarkdownStructuredParser:
    parser_id = "markdown-structured-v1"
    parser_version = "1.0.0"
    source_format = SourceFormat.MARKDOWN

    def parse(self, request: ParseRequest):
        started = time.monotonic()
        text, fallback_encoding = decode_text(request.content)
        blocks, tables, title = _markdown_blocks(text)
        warnings: list[ParseWarning] = []
        if fallback_encoding:
            warnings.append(
                ParseWarning(
                    code="INGEST_WARNING_TEXT_ENCODING_FALLBACK",
                    safe_message=f"Decoded text using {fallback_encoding}.",
                )
            )
        return complete_document(
            request=request,
            source_format=self.source_format,
            parser_name=self.parser_id,
            parser_version=self.parser_version,
            provider_name=request.route.provider,
            provider_version="python-stdlib",
            blocks=blocks,
            tables=tables,
            warnings=warnings,
            detected_title=title,
            detected_language=_language(text),
            normalized_markdown=text,
            started_at=started,
            reference_knowhere=True,
        )


class TextStructuredParser:
    parser_id = "text-structured-v1"
    parser_version = "1.0.0"
    source_format = SourceFormat.TEXT

    def parse(self, request: ParseRequest):
        started = time.monotonic()
        text, fallback_encoding = decode_text(request.content)
        blocks = [
            StructuredBlock(
                block_id=f"block_{index:04d}",
                kind=BlockKind.PARAGRAPH,
                text=normalized_text(paragraph),
            )
            for index, paragraph in enumerate(re.split(r"\n\s*\n", text), start=1)
            if normalized_text(paragraph)
        ]
        warnings = (
            (
                ParseWarning(
                    code="INGEST_WARNING_TEXT_ENCODING_FALLBACK",
                    safe_message=f"Decoded text using {fallback_encoding}.",
                ),
            )
            if fallback_encoding
            else ()
        )
        return complete_document(
            request=request,
            source_format=self.source_format,
            parser_name=self.parser_id,
            parser_version=self.parser_version,
            provider_name=request.route.provider,
            provider_version="python-stdlib",
            blocks=blocks,
            warnings=warnings,
            detected_language=_language(text),
            normalized_markdown="\n\n".join(block.text for block in blocks),
            started_at=started,
        )


class HtmlStructuredParser:
    parser_id = "html-structured-v1"
    parser_version = "1.0.0"
    source_format = SourceFormat.HTML

    def parse(self, request: ParseRequest):
        started = time.monotonic()
        text, fallback_encoding = decode_text(request.content)
        soup = BeautifulSoup(text, "lxml")
        for element in soup.find_all(
            ["script", "style", "nav", "header", "footer", "aside", "form"]
        ):
            element.decompose()
        container = soup.body or soup
        blocks: list[StructuredBlock] = []
        tables: list[TableAssetDraft] = []
        heading_levels: list[str] = []
        title = normalized_text(soup.title.get_text(" ")) if soup.title else None

        def current_path() -> tuple[str, ...]:
            return tuple(value for value in heading_levels if value)

        selected = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"}
        for element in container.find_all(selected):
            if not isinstance(element, Tag) or element.find_parent(selected):
                continue
            name = element.name.lower()
            value = normalized_text(element.get_text("\n" if name == "pre" else " "))
            if name.startswith("h"):
                level = int(name[1])
                heading_levels[level - 1 :] = []
                while len(heading_levels) < level:
                    heading_levels.append("")
                heading_levels[level - 1] = value
                title = title or value
                blocks.append(
                    StructuredBlock(
                        block_id=f"block_{len(blocks) + 1:04d}",
                        kind=BlockKind.HEADING,
                        text=value,
                        heading_path=current_path(),
                        location=SourceLocation(section_path=current_path()),
                        metadata={"level": level},
                    )
                )
            elif name == "table":
                rows = [
                    [normalized_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])]
                    for row in element.find_all("tr")
                ]
                rows = [row for row in rows if row]
                markdown, headers = matrix_to_markdown(rows)
                if not markdown:
                    continue
                asset_id = f"table_{len(tables) + 1:04d}"
                location = SourceLocation(section_path=current_path())
                tables.append(
                    TableAssetDraft(
                        asset_id=asset_id,
                        title=" / ".join(current_path()) or f"Table {len(tables) + 1}",
                        html=matrix_to_html(rows),
                        markdown=markdown,
                        headers=headers,
                        row_count=max(len(rows) - 1, 0),
                        column_count=max((len(row) for row in rows), default=0),
                        location=location,
                    )
                )
                blocks.append(
                    StructuredBlock(
                        block_id=f"block_{len(blocks) + 1:04d}",
                        kind=BlockKind.TABLE,
                        text=markdown,
                        heading_path=current_path(),
                        location=location,
                        table_asset_ids=(asset_id,),
                    )
                )
            elif value:
                kind = (
                    BlockKind.CODE
                    if name == "pre"
                    else (BlockKind.LIST_ITEM if name == "li" else BlockKind.PARAGRAPH)
                )
                blocks.append(
                    StructuredBlock(
                        block_id=f"block_{len(blocks) + 1:04d}",
                        kind=kind,
                        text=value,
                        heading_path=current_path(),
                        location=SourceLocation(section_path=current_path()),
                    )
                )

        warnings: list[ParseWarning] = []
        if fallback_encoding:
            warnings.append(
                ParseWarning(
                    code="INGEST_WARNING_TEXT_ENCODING_FALLBACK",
                    safe_message=f"Decoded text using {fallback_encoding}.",
                )
            )
        if container.find_all(["img", "iframe", "object", "embed"]):
            warnings.append(
                ParseWarning(
                    code="INGEST_WARNING_HTML_EXTERNAL_ASSETS_SKIPPED",
                    safe_message="HTML external or embedded resource references were not fetched.",
                )
            )
        markdown = markdownify_html(str(container), heading_style="ATX").strip()
        return complete_document(
            request=request,
            source_format=self.source_format,
            parser_name=self.parser_id,
            parser_version=self.parser_version,
            provider_name=request.route.provider,
            provider_version="beautifulsoup4+lxml+markdownify",
            blocks=blocks,
            tables=tables,
            warnings=warnings,
            detected_title=title,
            detected_language=_language(container.get_text(" ")),
            normalized_markdown=markdown,
            started_at=started,
        )


__all__: Sequence[str] = (
    "HtmlStructuredParser",
    "MarkdownStructuredParser",
    "TextStructuredParser",
)
