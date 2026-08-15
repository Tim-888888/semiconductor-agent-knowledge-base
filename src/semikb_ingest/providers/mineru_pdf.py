"""MinerU PDF provider isolated from application jobs, stores, and business models."""

from __future__ import annotations

import io
import json
import mimetypes
import posixpath
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import httpx

from semikb_ingest.errors import IngestError, IngestErrorCode
from semikb_ingest.providers.types import MinerUContentItem, MinerUImage, MinerUPdfResult


@dataclass(frozen=True, slots=True)
class MinerUPdfConfig:
    base_url: str
    api_key: str
    model_version: str = "vlm"
    timeout_seconds: float = 900
    poll_seconds: float = 3
    max_archive_entries: int = 20_000
    max_archive_uncompressed_bytes: int = 1024 * 1024 * 1024


class MinerUPdfClient:
    provider_name = "mineru"

    def __init__(
        self,
        config: MinerUPdfConfig,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not config.base_url.strip() or not config.api_key.strip():
            raise IngestError(
                IngestErrorCode.PARSER_NOT_CONFIGURED,
                "The MinerU endpoint and API key are required for PDF parsing.",
            )
        self.config = config
        self._transport = transport

    @property
    def provider_version(self) -> str:
        return self.config.model_version

    def parse_pdf(
        self,
        *,
        filename: str,
        content: bytes,
        correlation_id: str,
    ) -> MinerUPdfResult:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "X-SemiKB-Correlation-ID": correlation_id,
        }
        request_payload = {
            "files": [{"name": filename, "data_id": correlation_id}],
            "model_version": self.config.model_version,
            "language": "ch",
            "enable_table": True,
            "enable_formula": True,
        }
        deadline = time.monotonic() + self.config.timeout_seconds
        try:
            with httpx.Client(
                timeout=min(60, self.config.timeout_seconds),
                follow_redirects=True,
                transport=self._transport,
            ) as client:
                initial = client.post(
                    f"{self.config.base_url.rstrip('/')}/api/v4/file-urls/batch",
                    headers=headers,
                    json=request_payload,
                )
                data = self._validated_response(initial)
                upload_urls = data.get("file_urls", [])
                batch_id = data.get("batch_id")
                if not batch_id or not isinstance(upload_urls, list) or len(upload_urls) != 1:
                    self._raise_parse_failed("MinerU did not return one upload URL and batch ID.")
                upload = client.put(str(upload_urls[0]), content=content)
                upload.raise_for_status()
                archive_url = self._wait_for_archive(client, headers, str(batch_id), deadline)
                archive = client.get(archive_url)
                archive.raise_for_status()
        except IngestError:
            raise
        except httpx.TimeoutException as exc:
            raise IngestError(
                IngestErrorCode.PARSER_TIMEOUT,
                "MinerU PDF extraction timed out.",
            ) from exc
        except httpx.HTTPStatusError as exc:
            code = (
                IngestErrorCode.PARSER_UNAVAILABLE
                if exc.response.status_code == 429 or exc.response.status_code >= 500
                else IngestErrorCode.PARSE_FAILED
            )
            raise IngestError(code, "MinerU PDF extraction request failed.") from exc
        except httpx.HTTPError as exc:
            raise IngestError(
                IngestErrorCode.PARSER_UNAVAILABLE,
                "MinerU PDF extraction is unavailable.",
            ) from exc
        return self.read_archive(archive.content, self.config)

    def _wait_for_archive(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        batch_id: str,
        deadline: float,
    ) -> str:
        while time.monotonic() < deadline:
            response = client.get(
                f"{self.config.base_url.rstrip('/')}/api/v4/extract-results/batch/{batch_id}",
                headers=headers,
            )
            data = self._validated_response(response)
            raw = data.get("extract_result", [])
            records = raw if isinstance(raw, list) else [raw]
            if records and isinstance(records[0], dict):
                record = records[0]
                if record.get("state") == "done" and record.get("full_zip_url"):
                    return str(record["full_zip_url"])
                if record.get("state") == "failed":
                    self._raise_parse_failed("MinerU reported an extraction failure.")
            time.sleep(self.config.poll_seconds)
        raise IngestError(IngestErrorCode.PARSER_TIMEOUT, "MinerU PDF extraction timed out.")

    @staticmethod
    def _validated_response(response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 429 or response.status_code >= 500:
            raise IngestError(
                IngestErrorCode.PARSER_UNAVAILABLE,
                "MinerU PDF extraction is temporarily unavailable.",
            )
        if response.status_code >= 400:
            raise IngestError(
                IngestErrorCode.PARSE_FAILED,
                "MinerU rejected the PDF extraction request.",
            )
        try:
            result = response.json()
        except ValueError as exc:
            raise IngestError(
                IngestErrorCode.PARSE_FAILED,
                "MinerU returned an invalid response.",
            ) from exc
        if result.get("code") != 0 or not isinstance(result.get("data"), dict):
            raise IngestError(
                IngestErrorCode.PARSE_FAILED,
                "MinerU returned an unsuccessful extraction result.",
            )
        return result["data"]

    @classmethod
    def read_archive(
        cls,
        payload: bytes,
        config: MinerUPdfConfig | None = None,
    ) -> MinerUPdfResult:
        limits = config or MinerUPdfConfig(base_url="unused", api_key="unused")
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                infos = archive.infolist()
                if len(infos) > limits.max_archive_entries:
                    cls._raise_archive_limit()
                names: set[str] = set()
                total_size = 0
                for info in infos:
                    path = PurePosixPath(info.filename.replace("\\", "/"))
                    if (
                        path.is_absolute()
                        or ".." in path.parts
                        or (path.parts and path.parts[0].endswith(":"))
                    ):
                        raise IngestError(
                            IngestErrorCode.ASSET_EXTRACTION_FAILED,
                            "The MinerU archive contains an unsafe object path.",
                        )
                    normalized_name = path.as_posix()
                    if normalized_name in names:
                        raise IngestError(
                            IngestErrorCode.ASSET_EXTRACTION_FAILED,
                            "The MinerU archive contains duplicate object paths.",
                        )
                    names.add(normalized_name)
                    total_size += info.file_size
                    if total_size > limits.max_archive_uncompressed_bytes:
                        cls._raise_archive_limit()

                markdown_paths = sorted(name for name in names if name.endswith("full.md"))
                if not markdown_paths:
                    raise IngestError(
                        IngestErrorCode.EMPTY_PARSE_RESULT,
                        "The MinerU archive did not contain normalized Markdown.",
                    )
                markdown_path = markdown_paths[0]
                markdown = archive.read(markdown_path).decode("utf-8")
                content_paths = sorted(name for name in names if name.endswith("content_list.json"))
                raw_items: list[dict[str, Any]] = []
                content_base = posixpath.dirname(markdown_path)
                if content_paths:
                    content_path = content_paths[0]
                    parsed = json.loads(archive.read(content_path).decode("utf-8"))
                    if isinstance(parsed, list):
                        raw_items = [item for item in parsed if isinstance(item, dict)]
                    content_base = posixpath.dirname(content_path)

                items, referenced_images = cls._content_items(raw_items)
                markdown_images = {
                    target: caption.strip()
                    for caption, target in re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", markdown)
                    if not target.startswith(("http://", "https://", "data:"))
                }
                image_specs = dict(markdown_images)
                image_specs.update(referenced_images)
                images: list[MinerUImage] = []
                page_by_path = {
                    item.image_path: item.page_number
                    for item in items
                    if item.image_path is not None
                }
                for path, caption in image_specs.items():
                    normalized = posixpath.normpath(posixpath.join(content_base, path))
                    if normalized not in names:
                        alternate = posixpath.normpath(
                            posixpath.join(posixpath.dirname(markdown_path), path)
                        )
                        normalized = alternate if alternate in names else normalized
                    if normalized not in names:
                        continue
                    content_type = mimetypes.guess_type(normalized)[0] or "application/octet-stream"
                    if not content_type.startswith("image/"):
                        continue
                    images.append(
                        MinerUImage(
                            path=path,
                            filename=posixpath.basename(normalized),
                            content_type=content_type,
                            content=archive.read(normalized),
                            caption=caption,
                            page_number=page_by_path.get(path),
                        )
                    )
                pages = max((item.page_number or 0 for item in items), default=0)
                return MinerUPdfResult(
                    markdown=markdown,
                    content_items=tuple(items),
                    images=tuple(images),
                    pages=pages,
                )
        except IngestError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise IngestError(
                IngestErrorCode.CORRUPT_DOCUMENT,
                "The MinerU result archive is invalid.",
            ) from exc

    @staticmethod
    def _content_items(
        raw_items: list[dict[str, Any]],
    ) -> tuple[list[MinerUContentItem], dict[str, str]]:
        items: list[MinerUContentItem] = []
        image_paths: dict[str, str] = {}
        for raw in raw_items:
            raw_type = str(raw.get("type", "text"))
            page_idx = raw.get("page_idx")
            page_number = int(page_idx) + 1 if isinstance(page_idx, int) and page_idx >= 0 else None
            image_path = raw.get("img_path") if isinstance(raw.get("img_path"), str) else None
            caption_source = raw.get("image_caption") or raw.get("table_caption") or ""
            if isinstance(caption_source, list):
                caption = " ".join(
                    str(value).strip() for value in caption_source if str(value).strip()
                )
            else:
                caption = str(caption_source).strip()
            if image_path:
                image_paths[image_path] = caption
            text = str(raw.get("text") or raw.get("latex") or "").strip()
            level = raw.get("text_level")
            heading_level = int(level) if isinstance(level, int) and 1 <= level <= 6 else None
            kind = raw_type
            if raw_type in {"equation", "interline_equation"}:
                kind = "text"
            items.append(
                MinerUContentItem(
                    kind=kind,
                    text=text,
                    heading_level=heading_level,
                    page_number=page_number,
                    image_path=image_path,
                    table_html=str(raw.get("table_body") or ""),
                    table_caption=caption if raw_type == "table" else "",
                )
            )
        return items, image_paths

    @staticmethod
    def _raise_parse_failed(message: str) -> None:
        raise IngestError(IngestErrorCode.PARSE_FAILED, message)

    @staticmethod
    def _raise_archive_limit() -> None:
        raise IngestError(
            IngestErrorCode.DOCUMENT_LIMIT_EXCEEDED,
            "The MinerU result archive exceeds safe extraction limits.",
        )
