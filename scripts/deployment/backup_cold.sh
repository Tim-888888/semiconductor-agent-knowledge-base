#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" != "--apply" || "${T947_BACKUP_CONFIRM:-}" != "cold-backup-t947" ]]; then
  echo "Refusing cold backup. Use --apply with T947_BACKUP_CONFIRM=cold-backup-t947." >&2
  exit 2
fi

ENV_FILE="${SEMIKB_ENV_FILE:-$ROOT_DIR/.env}"
[[ "$ENV_FILE" = /* && -f "$ENV_FILE" ]] || {
  echo "SEMIKB_ENV_FILE must point to an existing absolute file." >&2
  exit 2
}

env_value() {
  awk -v key="$2" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); sub(/\r$/, ""); print; exit}' "$1"
}
data_root_value="$(env_value "$ENV_FILE" SEMIKB_DATA_ROOT)"
backup_root_value="$(env_value "$ENV_FILE" SEMIKB_BACKUP_ROOT)"
: "${data_root_value:?SEMIKB_DATA_ROOT is required}"
: "${backup_root_value:?SEMIKB_BACKUP_ROOT is required}"
SEMIKB_DATA_ROOT="$(realpath -m -- "$data_root_value")"
SEMIKB_BACKUP_ROOT="$(realpath -m -- "$backup_root_value")"
[[ "$SEMIKB_DATA_ROOT" != "/" && -d "$SEMIKB_DATA_ROOT" ]] || {
  echo "SEMIKB_DATA_ROOT must be an existing non-root directory." >&2
  exit 2
}
[[ "$SEMIKB_BACKUP_ROOT" != "/" && "$SEMIKB_BACKUP_ROOT" != "$SEMIKB_DATA_ROOT" ]] || {
  echo "SEMIKB_BACKUP_ROOT must be separate from SEMIKB_DATA_ROOT." >&2
  exit 2
}
case "$SEMIKB_BACKUP_ROOT/" in
  "$SEMIKB_DATA_ROOT/"*)
    echo "SEMIKB_BACKUP_ROOT cannot be inside SEMIKB_DATA_ROOT." >&2
    exit 2
    ;;
esac

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$SEMIKB_BACKUP_ROOT/semikb-$timestamp.tar.gz"
env_backup="$SEMIKB_BACKUP_ROOT/semikb-$timestamp.env"
manifest="$SEMIKB_BACKUP_ROOT/semikb-$timestamp.manifest.json"
install -d -m 0700 "$SEMIKB_BACKUP_ROOT"
umask 077

COMPOSE=(docker compose --env-file "$ENV_FILE" -f docker-compose.prod.yml)
stack_stopped=0
backup_complete=0

restart_stack() {
  if ((stack_stopped == 1)); then
    "${COMPOSE[@]}" up -d --wait --wait-timeout 300 >/dev/null
    stack_stopped=0
  fi
}

finish() {
  local exit_code=$?
  if ((backup_complete == 0)); then
    rm -f -- "$archive" "$archive.sha256" "$env_backup" "$env_backup.sha256" \
      "$manifest" "$manifest.sha256"
  fi
  restart_stack || true
  exit "$exit_code"
}
trap finish EXIT

stack_stopped=1
"${COMPOSE[@]}" stop web api worker
"${COMPOSE[@]}" stop mongodb redis milvus etcd minio milvus-minio
tar -C "$(dirname "$SEMIKB_DATA_ROOT")" -czf "$archive" "$(basename "$SEMIKB_DATA_ROOT")"
install -m 0600 "$ENV_FILE" "$env_backup"
sha256sum "$archive" > "$archive.sha256"
sha256sum "$env_backup" > "$env_backup.sha256"

python3 - "$archive" "$env_backup" "$manifest" "$SEMIKB_DATA_ROOT" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

archive = Path(sys.argv[1])
environment = Path(sys.argv[2])
manifest = Path(sys.argv[3])
data_root = Path(sys.argv[4])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = [path for path in data_root.rglob("*") if path.is_file()]
payload = {
    "schema": "semikb-t947-cold-backup-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_data_root": str(data_root),
    "source_file_count": len(files),
    "source_bytes": sum(path.stat().st_size for path in files),
    "archive": {
        "name": archive.name,
        "bytes": archive.stat().st_size,
        "sha256": sha256(archive),
    },
    "environment": {
        "name": environment.name,
        "sha256": sha256(environment),
        "mode": oct(environment.stat().st_mode & 0o777),
    },
}
manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.chmod(manifest, 0o600)
PY
sha256sum "$manifest" > "$manifest.sha256"
backup_complete=1
restart_stack
trap - EXIT
echo "Cold backup created: $archive"
echo "Matching protected environment backup created: $env_backup"
echo "Credential-safe backup manifest created: $manifest"
