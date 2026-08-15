"""Protocol implemented by semantic chunkers in the adapter stage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from semikb_ingest.models import ChunkDraft
from semikb_ingest.structure import StructuredBlock


class ChunkingStrategy(Protocol):
    @property
    def chunker_version(self) -> str: ...

    def chunk(self, blocks: Sequence[StructuredBlock]) -> tuple[ChunkDraft, ...]: ...
