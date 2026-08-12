"""Plan, build, or publish a governed Qwen embedding index migration."""

from __future__ import annotations

import argparse
import json

from semikb.config import Settings
from semikb.storage.embedding_index_migration import EmbeddingIndexMigrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index-version", required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--build",
        action="store_true",
        help="Build and validate the candidate without switching the active alias.",
    )
    action.add_argument(
        "--publish",
        action="store_true",
        help="Publish a previously built and externally evaluated candidate.",
    )
    args = parser.parse_args()
    settings = Settings(demo_mode=False)
    migrator = EmbeddingIndexMigrator(settings, args.target_index_version)
    if args.build:
        result = migrator.build()
    elif args.publish:
        result = migrator.publish()
    else:
        result = {"status": "dry_run", "plan": migrator.plan().as_dict()}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
