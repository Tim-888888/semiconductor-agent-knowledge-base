"""Document parser protocols and explicit adapter registry."""

from semikb_ingest.parsers.docx import DocxStructuredParser
from semikb_ingest.parsers.image import ImageVlmParser
from semikb_ingest.parsers.pdf import PdfMinerUParser
from semikb_ingest.parsers.pptx import PptxStructuredParser
from semikb_ingest.parsers.registry import (
    DocumentParser,
    ParseRequest,
    ParserRegistry,
)
from semikb_ingest.parsers.tabular import CsvStructuredParser, XlsxStructuredParser
from semikb_ingest.parsers.textual import (
    HtmlStructuredParser,
    MarkdownStructuredParser,
    TextStructuredParser,
)

__all__ = [
    "CsvStructuredParser",
    "DocumentParser",
    "DocxStructuredParser",
    "HtmlStructuredParser",
    "ImageVlmParser",
    "MarkdownStructuredParser",
    "ParseRequest",
    "ParserRegistry",
    "PdfMinerUParser",
    "PptxStructuredParser",
    "TextStructuredParser",
    "XlsxStructuredParser",
]
