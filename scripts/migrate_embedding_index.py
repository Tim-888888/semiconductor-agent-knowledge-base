"""Plan or apply the governed Qwen embedding index rebuild."""

from __future__ import annotations

import argparse
import json

from semikb.config import Settings
from semikb.storage.embedding_index_migration import EmbeddingIndexMigrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index-version", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create, validate, and activate the new index. The old collection is retained.",
    )
    args = parser.parse_args()
    settings = Settings(demo_mode=False)
    migrator = EmbeddingIndexMigrator(settings, args.target_index_version)
    result = migrator.apply() if args.apply else {"status": "dry_run", "plan": migrator.plan().as_dict()}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
