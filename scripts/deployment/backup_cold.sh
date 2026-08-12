#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
env_value() {
  awk -v key="$1" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); sub(/\r$/, ""); print; exit}' .env
}
SEMIKB_DATA_ROOT="$(env_value SEMIKB_DATA_ROOT)"
SEMIKB_BACKUP_ROOT="$(env_value SEMIKB_BACKUP_ROOT)"
: "${SEMIKB_DATA_ROOT:?SEMIKB_DATA_ROOT is required}"
: "${SEMIKB_BACKUP_ROOT:?SEMIKB_BACKUP_ROOT is required}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
archive="$SEMIKB_BACKUP_ROOT/semikb-$timestamp.tar.gz"
env_backup="$SEMIKB_BACKUP_ROOT/semikb-$timestamp.env"
mkdir -p "$SEMIKB_BACKUP_ROOT"

restart_stack() {
  docker compose --env-file .env -f docker-compose.prod.yml up -d >/dev/null
}
trap restart_stack EXIT
docker compose --env-file .env -f docker-compose.prod.yml stop web api worker mongodb redis milvus etcd minio milvus-minio
tar -C "$(dirname "$SEMIKB_DATA_ROOT")" -czf "$archive" "$(basename "$SEMIKB_DATA_ROOT")"
install -m 0600 .env "$env_backup"
sha256sum "$archive" > "$archive.sha256"
sha256sum "$env_backup" > "$env_backup.sha256"
trap - EXIT
restart_stack
echo "Cold backup created: $archive"
echo "Matching protected environment backup created: $env_backup"
