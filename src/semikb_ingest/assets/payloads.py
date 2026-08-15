"""In-memory payload handles that cannot be mistaken for durable object references."""

from __future__ import annotations

import hashlib
import threading
import uuid

from semikb_ingest.models import BinaryPayload


class ProcessPayloadStore:
    """Own extracted bytes until the governed ingestion layer persists them."""

    def __init__(self) -> None:
        self._payloads: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def put(self, filename: str, content_type: str, content: bytes) -> BinaryPayload:
        if not content:
            raise ValueError("Asset payload cannot be empty.")
        handle = f"payload_{uuid.uuid4().hex}"
        payload = BinaryPayload(
            handle=handle,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        with self._lock:
            self._payloads[handle] = bytes(content)
        return payload

    def read(self, payload: BinaryPayload) -> bytes:
        with self._lock:
            content = self._payloads.get(payload.handle)
        if content is None:
            raise KeyError(payload.handle)
        self._verify(payload, content)
        return content

    def pop(self, payload: BinaryPayload) -> bytes:
        with self._lock:
            content = self._payloads.pop(payload.handle, None)
        if content is None:
            raise KeyError(payload.handle)
        self._verify(payload, content)
        return content

    def discard(self, handle: str) -> None:
        with self._lock:
            self._payloads.pop(handle, None)

    @staticmethod
    def _verify(payload: BinaryPayload, content: bytes) -> None:
        if len(content) != payload.size_bytes:
            raise ValueError("Asset payload size no longer matches its handle metadata.")
        if hashlib.sha256(content).hexdigest() != payload.sha256:
            raise ValueError("Asset payload hash no longer matches its handle metadata.")
