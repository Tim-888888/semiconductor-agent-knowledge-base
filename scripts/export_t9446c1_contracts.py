"""Export the versioned T9-4.4.6c-1 generalization contract bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from semikb.contracts.models import (
    ActorScope,
    Chunk,
    DocumentRevision,
    EvaluationDataset,
    EvaluationRun,
    ImageAsset,
    IngestDocumentRequest,
    IngestUploadMetadata,
    RetrievalConstraints,
    SourceManifest,
    TableAsset,
)

CONTRACT_VERSION = "semikb-generalization-baseline-v1"
DEFAULT_OUTPUT = Path(
    "docs/evidence/t9-4-4-6c-1/generalization-contract-v1.schema.json"
)
CONTRACT_MODELS: tuple[type[BaseModel], ...] = (
    ActorScope,
    SourceManifest,
    DocumentRevision,
    Chunk,
    ImageAsset,
    TableAsset,
    IngestDocumentRequest,
    IngestUploadMetadata,
    RetrievalConstraints,
    EvaluationDataset,
    EvaluationRun,
)


def render_contract_bundle() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "schemas": {model.__name__: model.model_json_schema() for model in CONTRACT_MODELS},
    }


def export_contracts(output: Path = DEFAULT_OUTPUT, *, check: bool = False) -> dict[str, Any]:
    bundle = render_contract_bundle()
    if check:
        if not output.is_file():
            raise FileNotFoundError(output)
        if json.loads(output.read_text(encoding="utf-8")) != bundle:
            raise ValueError(f"Versioned contract bundle is stale: {output}")
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
    args = parser.parse_args()
    print(json.dumps(export_contracts(args.output, check=args.check), indent=2))


if __name__ == "__main__":
    main()
