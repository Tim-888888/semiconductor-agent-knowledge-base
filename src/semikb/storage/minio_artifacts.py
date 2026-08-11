"""Private MinIO object persistence for replayable ingestion artifacts."""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path
from urllib.parse import quote

from semikb.contracts.models import ObjectRef
from semikb.storage.clients import StorageClientFactory

_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _segment(value: str, field: str) -> str:
    if not value or not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(f"Unsafe {field}; use letters, numbers, dot, underscore, or hyphen.")
    return value


class MinioArtifactRepository:
    """Stores immutable source and derived artifacts under deterministic keys."""

    def __init__(self, factory: StorageClientFactory) -> None:
        self._factory = factory

    def store_source(
        self,
        *,
        document_id: str,
        revision: str,
        filename: str,
        content: bytes,
        content_type: str,
        source_hash: str,
    ) -> ObjectRef:
        if hashlib.sha256(content).hexdigest() != source_hash:
            raise ValueError("Source content does not match its SHA-256 digest.")
        source_name = Path(filename).name
        if not source_name:
            raise ValueError("Source filename is required.")
        safe_name = quote(source_name, safe="._-")
        key = (
            f"documents/{_segment(document_id, 'document_id')}/"
            f"{_segment(revision, 'revision')}/source/{source_hash}/{safe_name}"
        )
        return self._put(
            "semikb-raw",
            key,
            content,
            content_type,
            source_hash,
            document_id=document_id,
            revision=revision,
            content_class="source",
        )

    def store_parsed_markdown(
        self,
        *,
        document_id: str,
        revision: str,
        parser_version: str,
        source_hash: str,
        content: bytes,
    ) -> ObjectRef:
        key = (
            f"documents/{_segment(document_id, 'document_id')}/"
            f"{_segment(revision, 'revision')}/parse/"
            f"{_segment(parser_version, 'parser_version')}/{source_hash}/document.md"
        )
        return self._put(
            "semikb-derived",
            key,
            content,
            "text/markdown",
            hashlib.sha256(content).hexdigest(),
            document_id=document_id,
            revision=revision,
            content_class="parsed_markdown",
        )

    def store_image_asset(
        self,
        *,
        document_id: str,
        revision: str,
        image_id: str,
        filename: str,
        content: bytes,
        content_type: str,
        source_hash: str,
    ) -> ObjectRef:
        suffix = Path(filename).suffix.lower() or ".bin"
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            raise ValueError("Unsafe image filename extension.")
        key = (
            f"documents/{_segment(document_id, 'document_id')}/"
            f"{_segment(revision, 'revision')}/assets/"
            f"{_segment(image_id, 'image_id')}/original{suffix}"
        )
        return self._put(
            "semikb-derived",
            key,
            content,
            content_type,
            hashlib.sha256(content).hexdigest(),
            document_id=document_id,
            revision=revision,
            asset_id=image_id,
            source_sha256=source_hash,
            content_class="image",
        )

    def load_object(self, object_ref: ObjectRef) -> bytes:
        client = self._factory.create_minio()
        response = client.get_object(object_ref.bucket, object_ref.object_key)
        try:
            content = response.read()
        finally:
            response.close()
            response.release_conn()
        if hashlib.sha256(content).hexdigest() != object_ref.sha256:
            raise ValueError("Stored object failed SHA-256 verification.")
        return content

    def _put(
        self,
        bucket: str,
        object_key: str,
        content: bytes,
        content_type: str,
        sha256: str,
        **metadata: str,
    ) -> ObjectRef:
        client = self._factory.create_minio()
        result = client.put_object(
            bucket,
            object_key,
            io.BytesIO(content),
            length=len(content),
            content_type=content_type,
            metadata={**metadata, "schema_version": "1", "sha256": sha256},
        )
        return ObjectRef(
            bucket=bucket,
            object_key=object_key,
            content_type=content_type,
            sha256=sha256,
            version_id=getattr(result, "version_id", None),
        )
