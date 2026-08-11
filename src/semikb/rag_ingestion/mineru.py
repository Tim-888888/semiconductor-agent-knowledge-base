"""MinerU Precision Extract adapter for long-running Celery ingestion workers.

MinerU's Token-based API uses signed upload URLs, then asynchronous batch polling.
This module returns only normalized Markdown and deliberately owns no business state.
"""

from __future__ import annotations

import io
import time
import zipfile
from typing import Any

import httpx

from semikb.config import Settings


class MinerUError(RuntimeError):
    """A safe parser error suitable for an ingestion job event."""


class MinerUPrecisionClient:
    """Calls the current Token-based MinerU precision API without leaking its protocol."""

    def __init__(self, settings: Settings) -> None:
        if not settings.mineru_api_base_url or not settings.mineru_api_key:
            raise MinerUError("MinerU endpoint and API key are required for binary document parsing.")
        self._base_url = settings.mineru_api_base_url.rstrip("/")
        self._settings = settings

    def parse_file(self, filename: str, content: bytes, data_id: str) -> str:
        headers = {"Authorization": f"Bearer {self._settings.mineru_api_key}"}
        request_payload = {
            "files": [{"name": filename, "data_id": data_id}],
            "model_version": self._settings.mineru_model_version,
            "language": "ch",
            "enable_table": True,
            "enable_formula": True,
        }
        deadline = time.monotonic() + self._settings.mineru_timeout_seconds
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.post(
                f"{self._base_url}/api/v4/file-urls/batch",
                headers=headers,
                json=request_payload,
            )
            result = self._validated_json(response)
            data = result["data"]
            upload_urls = data.get("file_urls", [])
            batch_id = data.get("batch_id")
            if not batch_id or len(upload_urls) != 1:
                raise MinerUError("MinerU did not return one upload URL and a batch identifier.")
            uploaded = client.put(upload_urls[0], content=content, headers={"Content-Type": "application/octet-stream"})
            uploaded.raise_for_status()
            zip_url = self._wait_for_result(client, headers, str(batch_id), deadline)
            archive = client.get(zip_url)
            archive.raise_for_status()
        return self._read_full_markdown(archive.content)

    def _wait_for_result(
        self,
        client: httpx.Client,
        headers: dict[str, str],
        batch_id: str,
        deadline: float,
    ) -> str:
        while time.monotonic() < deadline:
            response = client.get(f"{self._base_url}/api/v4/extract-results/batch/{batch_id}", headers=headers)
            result = self._validated_json(response)
            raw_results = result["data"].get("extract_result", [])
            records = raw_results if isinstance(raw_results, list) else [raw_results]
            if not records:
                time.sleep(self._settings.mineru_poll_seconds)
                continue
            record = records[0]
            state = record.get("state")
            if state == "done" and record.get("full_zip_url"):
                return str(record["full_zip_url"])
            if state == "failed":
                raise MinerUError("MinerU extraction failed: " + str(record.get("err_msg", "unknown error"))[:300])
            time.sleep(self._settings.mineru_poll_seconds)
        raise MinerUError("MinerU extraction timed out; retry the ingestion job later.")

    @staticmethod
    def _validated_json(response: httpx.Response) -> dict[str, Any]:
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise MinerUError("MinerU request rejected: " + str(result.get("msg", "unknown error"))[:300])
        if not isinstance(result.get("data"), dict):
            raise MinerUError("MinerU response did not contain an expected data object.")
        return result

    @staticmethod
    def _read_full_markdown(payload: bytes) -> str:
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                markdown_paths = [name for name in archive.namelist() if name.endswith("full.md")]
                if not markdown_paths:
                    raise MinerUError("MinerU archive did not contain full.md.")
                return archive.read(markdown_paths[0]).decode("utf-8")
        except zipfile.BadZipFile as exc:
            raise MinerUError("MinerU result was not a valid ZIP archive.") from exc
