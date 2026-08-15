"""Export and verify the frozen T9-4.4.6a governance contract bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from semikb.contracts.models import (
    DocumentLifecycleOperationRecord,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentRevisionSummary,
    RestoreDocumentRevisionRequest,
    SourceManifest,
    WithdrawDocumentRevisionRequest,
)

CONTRACT_VERSION = "semikb-source-governance-v1"
DEFAULT_OUTPUT = Path("docs/evidence/t9-4-4-6a/source-governance-contract-v1.schema.json")
DEFAULT_MANIFEST_DIRECTORY = Path("data/source_manifests")
CONTRACT_MODELS: tuple[type[BaseModel], ...] = (
    SourceManifest,
    KnowledgeDocumentListResponse,
    KnowledgeDocumentRevisionSummary,
    WithdrawDocumentRevisionRequest,
    RestoreDocumentRevisionRequest,
    DocumentLifecycleOperationRecord,
)


def render_contract_bundle() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "schemas": {model.__name__: model.model_json_schema() for model in CONTRACT_MODELS},
    }


def validate_source_manifests(
    directory: Path = DEFAULT_MANIFEST_DIRECTORY,
    *,
    repository_root: Path = Path("."),
) -> dict[str, Any]:
    manifests: list[SourceManifest] = []
    verified_hashes: list[str] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(directory.glob("*.json")):
        manifest = SourceManifest.model_validate_json(path.read_text(encoding="utf-8"))
        key = (manifest.source_id, manifest.manifest_version)
        if key in seen:
            raise ValueError(f"Duplicate source manifest version: {key[0]}:{key[1]}")
        seen.add(key)
        scoped_path = repository_root / manifest.hash_scope
        if scoped_path.is_file():
            actual_hash = hashlib.sha256(scoped_path.read_bytes()).hexdigest()
            if actual_hash != manifest.source_hash.lower():
                raise ValueError(
                    f"Source hash mismatch for {path}: expected {manifest.source_hash}, "
                    f"got {actual_hash}"
                )
            verified_hashes.append(str(scoped_path.as_posix()))
        manifests.append(manifest)
    if not manifests:
        raise ValueError(f"No source manifests found in {directory}")
    return {
        "manifest_count": len(manifests),
        "source_versions": [f"{item.source_id}:{item.manifest_version}" for item in manifests],
        "locally_verified_hashes": verified_hashes,
    }


def export_contracts(output: Path = DEFAULT_OUTPUT, *, check: bool = False) -> dict[str, Any]:
    bundle = render_contract_bundle()
    if check:
        if not output.is_file():
            raise FileNotFoundError(output)
        current = json.loads(output.read_text(encoding="utf-8"))
        if current != bundle:
            raise ValueError(f"Frozen contract bundle is stale: {output}")
        status = "current"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        status = "written"
    return {
        "contract_version": CONTRACT_VERSION,
        "schema_count": len(CONTRACT_MODELS),
        "output": str(output),
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-directory", type=Path, default=DEFAULT_MANIFEST_DIRECTORY)
    args = parser.parse_args()
    result = {
        "contracts": export_contracts(args.output, check=args.check),
        "manifests": validate_source_manifests(args.manifest_directory),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
