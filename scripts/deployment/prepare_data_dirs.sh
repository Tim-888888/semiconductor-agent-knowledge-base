#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/.env}"
[[ -f "$ENV_FILE" ]] || { echo "Missing environment file: $ENV_FILE" >&2; exit 1; }
env_value() {
  awk -v key="$1" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); sub(/\r$/, ""); print; exit}' "$ENV_FILE"
}
SEMIKB_DATA_ROOT="$(env_value SEMIKB_DATA_ROOT)"
SEMIKB_BACKUP_ROOT="$(env_value SEMIKB_BACKUP_ROOT)"
: "${SEMIKB_DATA_ROOT:?SEMIKB_DATA_ROOT is required}"
: "${SEMIKB_BACKUP_ROOT:?SEMIKB_BACKUP_ROOT is required}"

install -d -m 0750 \
  "$SEMIKB_DATA_ROOT/mongodb" \
  "$SEMIKB_DATA_ROOT/redis" \
  "$SEMIKB_DATA_ROOT/minio" \
  "$SEMIKB_DATA_ROOT/milvus/etcd" \
  "$SEMIKB_DATA_ROOT/milvus/minio" \
  "$SEMIKB_DATA_ROOT/milvus/data" \
  "$SEMIKB_BACKUP_ROOT"
echo "Project-owned data directories are ready."
