"""Explicit, signal-validated routing for supported document formats."""

from __future__ import annotations

import io
import mimetypes
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.models import SourceFormat

GENERIC_MEDIA_TYPES = {
    "",
    "application/octet-stream",
    "application/x-zip-compressed",
    "application/zip",
    "binary/octet-stream",
}
ZIP_PREFIXES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    max_source_bytes: int = 100 * 1024 * 1024
    max_office_entries: int = 10_000
    max_office_uncompressed_bytes: int = 512 * 1024 * 1024
    max_office_compression_ratio: float = 100.0


@dataclass(frozen=True, slots=True)
class ExtensionRule:
    extension: str
    media_types: frozenset[str]
    signatures: tuple[str, ...] = ()
    office_family: SourceFormat | None = None


@dataclass(frozen=True, slots=True)
class FormatRoute:
    source_format: SourceFormat
    parser_id: str
    provider: str
    fallback: str
    rules: tuple[ExtensionRule, ...]


@dataclass(frozen=True, slots=True)
class ResolvedRoute:
    source_format: SourceFormat
    parser_id: str
    provider: str
    fallback: str
    normalized_extension: str
    declared_media_type: str
    detected_signature: str | None


FORMAT_ROUTES: tuple[FormatRoute, ...] = (
    FormatRoute(
        SourceFormat.MARKDOWN,
        "markdown-structured-v1",
        "builtin",
        "none",
        (
            ExtensionRule(".md", frozenset({"text/markdown", "text/plain"})),
            ExtensionRule(".markdown", frozenset({"text/markdown", "text/plain"})),
        ),
    ),
    FormatRoute(
        SourceFormat.TEXT,
        "text-structured-v1",
        "builtin",
        "none",
        (ExtensionRule(".txt", frozenset({"text/plain"})),),
    ),
    FormatRoute(
        SourceFormat.HTML,
        "html-structured-v1",
        "local-html",
        "none",
        (
            ExtensionRule(".html", frozenset({"text/html"})),
            ExtensionRule(".htm", frozenset({"text/html"})),
        ),
    ),
    FormatRoute(
        SourceFormat.PDF,
        "pdf-mineru-v1",
        "mineru",
        "none",
        (ExtensionRule(".pdf", frozenset({"application/pdf"}), ("pdf",)),),
    ),
    FormatRoute(
        SourceFormat.DOCX,
        "docx-structured-v1",
        "local-docx",
        "none",
        (
            ExtensionRule(
                ".docx",
                frozenset(
                    {
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    }
                ),
                ("zip",),
                SourceFormat.DOCX,
            ),
        ),
    ),
    FormatRoute(
        SourceFormat.XLSX,
        "xlsx-structured-v1",
        "local-xlsx",
        "none",
        (
            ExtensionRule(
                ".xlsx",
                frozenset(
                    {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
                ),
                ("zip",),
                SourceFormat.XLSX,
            ),
        ),
    ),
    FormatRoute(
        SourceFormat.CSV,
        "csv-structured-v1",
        "builtin-csv",
        "none",
        (
            ExtensionRule(
                ".csv",
                frozenset({"text/csv", "text/plain", "application/csv"}),
            ),
        ),
    ),
    FormatRoute(
        SourceFormat.PPTX,
        "pptx-structured-v1",
        "local-pptx",
        "none",
        (
            ExtensionRule(
                ".pptx",
                frozenset(
                    {
                        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                    }
                ),
                ("zip",),
                SourceFormat.PPTX,
            ),
        ),
    ),
    FormatRoute(
        SourceFormat.IMAGE,
        "image-vlm-v1",
        "qwen-vl",
        "ocr_only_when_configured",
        (
            ExtensionRule(".png", frozenset({"image/png"}), ("png",)),
            ExtensionRule(".jpg", frozenset({"image/jpeg"}), ("jpeg",)),
            ExtensionRule(".jpeg", frozenset({"image/jpeg"}), ("jpeg",)),
        ),
    ),
)

_EXTENSION_INDEX = {
    rule.extension: (route, rule) for route in FORMAT_ROUTES for rule in route.rules
}
_OFFICE_REQUIRED_MEMBERS = {
    SourceFormat.DOCX: frozenset({"[Content_Types].xml", "word/document.xml"}),
    SourceFormat.XLSX: frozenset({"[Content_Types].xml", "xl/workbook.xml"}),
    SourceFormat.PPTX: frozenset({"[Content_Types].xml", "ppt/presentation.xml"}),
}


class FormatRouter:
    """Resolve one declared format without any cross-format parser fallback."""

    def __init__(self, policy: RoutingPolicy | None = None) -> None:
        self.policy = policy or RoutingPolicy()

    def resolve(
        self,
        filename: str,
        content: bytes,
        declared_media_type: str | None = None,
    ) -> ResolvedRoute:
        if not content:
            raise IngestError(
                IngestErrorCode.CORRUPT_DOCUMENT,
                "The uploaded document is empty.",
            )
        if len(content) > self.policy.max_source_bytes:
            raise IngestError(
                IngestErrorCode.DOCUMENT_LIMIT_EXCEEDED,
                "The uploaded document exceeds the configured size limit.",
            )

        extension = Path(filename).suffix.lower()
        media_type = self._normalize_media_type(declared_media_type)
        signature = self._detect_signature(content)
        office_family = self._inspect_office(content) if signature == "zip" else None

        if extension:
            indexed = _EXTENSION_INDEX.get(extension)
            if indexed is None:
                raise IngestError(
                    IngestErrorCode.UNSUPPORTED_FORMAT,
                    f"The file extension {extension!r} is not supported.",
                )
            route, rule = indexed
        else:
            route, rule = self._resolve_without_extension(media_type, signature, office_family)

        self._validate_media_type(rule, media_type)
        self._validate_signature(rule, signature)
        if rule.office_family is not None:
            if office_family is None:
                raise IngestError(
                    IngestErrorCode.INVALID_OFFICE_CONTAINER,
                    "The Office document container is invalid or incomplete.",
                )
            if office_family is not rule.office_family:
                raise IngestError(
                    IngestErrorCode.FILE_TYPE_MISMATCH,
                    "The Office container does not match the file extension or media type.",
                )

        return ResolvedRoute(
            source_format=route.source_format,
            parser_id=route.parser_id,
            provider=route.provider,
            fallback=route.fallback,
            normalized_extension=extension,
            declared_media_type=media_type,
            detected_signature=signature,
        )

    def _resolve_without_extension(
        self,
        media_type: str,
        signature: str | None,
        office_family: SourceFormat | None,
    ) -> tuple[FormatRoute, ExtensionRule]:
        candidates: list[tuple[FormatRoute, ExtensionRule]] = []
        for route in FORMAT_ROUTES:
            for rule in route.rules:
                media_matches = media_type not in GENERIC_MEDIA_TYPES and media_type in rule.media_types
                signature_matches = signature is not None and signature in rule.signatures
                office_matches = office_family is not None and office_family is rule.office_family
                if rule.office_family is not None:
                    if office_matches and (
                        media_type in GENERIC_MEDIA_TYPES or media_matches
                    ):
                        candidates.append((route, rule))
                elif media_matches and (not rule.signatures or signature_matches):
                    candidates.append((route, rule))
                elif (
                    signature_matches
                    and media_type in GENERIC_MEDIA_TYPES
                ):
                    candidates.append((route, rule))

        unique = {(route.source_format, rule.office_family, tuple(rule.signatures)) for route, rule in candidates}
        if len(unique) != 1:
            known_media_type = any(
                media_type in rule.media_types
                for route in FORMAT_ROUTES
                for rule in route.rules
            )
            code = IngestErrorCode.UNSUPPORTED_FORMAT
            if not candidates and known_media_type:
                code = IngestErrorCode.FILE_TYPE_MISMATCH
            raise IngestError(code, "The document format could not be resolved unambiguously.")
        return candidates[0]

    @staticmethod
    def _normalize_media_type(media_type: str | None) -> str:
        if not media_type:
            return ""
        return media_type.split(";", maxsplit=1)[0].strip().lower()

    @staticmethod
    def _detect_signature(content: bytes) -> str | None:
        if content.startswith(b"%PDF-"):
            return "pdf"
        if content.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if content.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if content.startswith(ZIP_PREFIXES):
            return "zip"
        return None

    @staticmethod
    def _validate_media_type(rule: ExtensionRule, media_type: str) -> None:
        if media_type in GENERIC_MEDIA_TYPES:
            return
        if media_type not in rule.media_types:
            raise IngestError(
                IngestErrorCode.FILE_TYPE_MISMATCH,
                "The declared media type does not match the detected file format.",
            )

    @staticmethod
    def _validate_signature(rule: ExtensionRule, signature: str | None) -> None:
        if rule.signatures:
            if signature not in rule.signatures:
                raise IngestError(
                    IngestErrorCode.FILE_TYPE_MISMATCH,
                    "The file signature does not match the declared document format.",
                )
        elif signature is not None:
            raise IngestError(
                IngestErrorCode.FILE_TYPE_MISMATCH,
                "Binary content cannot be routed through a text document adapter.",
            )

    def _inspect_office(self, content: bytes) -> SourceFormat | None:
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                infos = archive.infolist()
                if len(infos) > self.policy.max_office_entries:
                    self._raise_zip_bomb()
                total_uncompressed = 0
                names: set[str] = set()
                for info in infos:
                    normalized = PurePosixPath(info.filename.replace("\\", "/"))
                    if (
                        normalized.is_absolute()
                        or ".." in normalized.parts
                        or (normalized.parts and normalized.parts[0].endswith(":"))
                    ):
                        raise IngestError(
                            IngestErrorCode.INVALID_OFFICE_CONTAINER,
                            "The Office container contains an unsafe object path.",
                        )
                    if info.flag_bits & 0x1:
                        raise IngestError(
                            IngestErrorCode.ENCRYPTED_DOCUMENT,
                            "Encrypted Office documents are not supported.",
                        )
                    total_uncompressed += info.file_size
                    if total_uncompressed > self.policy.max_office_uncompressed_bytes:
                        self._raise_zip_bomb()
                    if info.file_size:
                        if info.compress_size == 0:
                            self._raise_zip_bomb()
                        ratio = info.file_size / info.compress_size
                        if ratio > self.policy.max_office_compression_ratio:
                            self._raise_zip_bomb()
                    normalized_name = normalized.as_posix()
                    if normalized_name in names:
                        raise IngestError(
                            IngestErrorCode.INVALID_OFFICE_CONTAINER,
                            "The Office container contains duplicate object paths.",
                        )
                    names.add(normalized_name)
        except IngestError:
            raise
        except zipfile.BadZipFile as exc:
            raise IngestError(
                IngestErrorCode.INVALID_OFFICE_CONTAINER,
                "The Office document is not a valid ZIP container.",
            ) from exc

        matches = [
            family
            for family, required in _OFFICE_REQUIRED_MEMBERS.items()
            if required.issubset(names)
        ]
        if len(matches) > 1:
            raise IngestError(
                IngestErrorCode.INVALID_OFFICE_CONTAINER,
                "The Office container contains conflicting document families.",
            )
        return matches[0] if matches else None

    @staticmethod
    def _raise_zip_bomb() -> None:
        raise IngestError(
            IngestErrorCode.ZIP_BOMB_SUSPECTED,
            "The Office container exceeds safe archive expansion limits.",
        )


def route_table() -> tuple[FormatRoute, ...]:
    """Return the immutable format registry for audits and tests."""

    return FORMAT_ROUTES


def guessed_media_type(filename: str) -> str:
    """Return a filename hint; callers must still pass content through FormatRouter."""

    return mimetypes.guess_type(filename)[0] or "application/octet-stream"
