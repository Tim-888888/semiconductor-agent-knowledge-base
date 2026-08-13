"""Plan, apply, or roll back the T9-4.3.1 MongoDB context migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from semikb.config import get_settings
from semikb.storage.t9431_context_migration import migrate, rollback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true")
    action.add_argument("--rollback", type=Path)
    parser.add_argument("--snapshot-path", type=Path)
    args = parser.parse_args()
    if args.rollback:
        result = rollback(get_settings(), args.rollback)
    else:
        result = migrate(
            get_settings(),
            apply=args.apply,
            snapshot_path=args.snapshot_path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
