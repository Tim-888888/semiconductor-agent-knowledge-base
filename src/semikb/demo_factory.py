"""Explicit demonstration identities and governance values.

Public contracts stay neutral. Demo-only values enter the system through this
module so they cannot silently label an unknown production upload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from semikb.contracts.models import (
    ActorScope,
    RetrievalPolicy,
    SourceManifest,
)


def demo_actor_scope(
    *,
    user_id: str = "demo_engineer",
    roles: list[str] | None = None,
) -> ActorScope:
    return ActorScope(
        user_id=user_id,
        roles=roles or ["engineer"],
        access_scope_keys=["demo_engineering"],
        fabs=["FAB-01"],
        products=["P-ALPHA"],
        tool_ids=["ETCH-03"],
    )


def load_demo_source_manifest(path: Path) -> SourceManifest:
    return SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))


def apply_demo_ingestion_governance(
    payload: dict[str, Any],
    manifest: SourceManifest,
) -> dict[str, Any]:
    """Attach explicit fixture governance without changing the source payload."""

    document_type = str(payload.get("document_type", "")).strip().lower()
    return {
        **payload,
        "source_id": manifest.source_id,
        "source_manifest_version": manifest.manifest_version,
        "dataset_version": manifest.dataset_version,
        "source_license_status": manifest.license_status.value,
        "redistribution_policy": manifest.redistribution_policy.value,
        "access_scope_key": manifest.access_scope_key,
        "retrieval_policy": (
            RetrievalPolicy.PROTECTED.value
            if document_type == "sop"
            else RetrievalPolicy.STANDARD.value
        ),
    }
