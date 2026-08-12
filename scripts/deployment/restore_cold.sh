#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--archive" || -z "${2:-}" || "${3:-}" != "--env" || -z "${4:-}" || "${5:-}" != "--confirm-empty-target" ]]; then
  echo "Usage: $0 --archive /absolute/backup.tar.gz --env /absolute/backup.env --confirm-empty-target" >&2
  exit 2
fi
archive="$2"
env_backup="$4"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
[[ -f "$env_backup" && -f "$env_backup.sha256" ]] || { echo "Environment backup or checksum is missing." >&2; exit 1; }
expected_env_hash="$(awk 'NR==1 {print $1}' "$env_backup.sha256")"
actual_env_hash="$(sha256sum "$env_backup" | awk '{print $1}')"
[[ "$actual_env_hash" == "$expected_env_hash" ]] || { echo "Environment checksum validation failed." >&2; exit 1; }
env_value() {
  awk -v key="$1" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); sub(/\r$/, ""); print; exit}' "$env_backup"
}
SEMIKB_DATA_ROOT="$(env_value SEMIKB_DATA_ROOT)"
: "${SEMIKB_DATA_ROOT:?SEMIKB_DATA_ROOT is required}"
[[ -f "$archive" && -f "$archive.sha256" ]] || { echo "Archive or checksum is missing." >&2; exit 1; }
expected_hash="$(awk 'NR==1 {print $1}' "$archive.sha256")"
actual_hash="$(sha256sum "$archive" | awk '{print $1}')"
[[ "$actual_hash" == "$expected_hash" ]] || { echo "Archive checksum validation failed." >&2; exit 1; }

docker compose --env-file "$env_backup" -f docker-compose.prod.yml down
if [[ -d "$SEMIKB_DATA_ROOT" ]] && find "$SEMIKB_DATA_ROOT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing restore because target is not empty: $SEMIKB_DATA_ROOT" >&2
  exit 1
fi
mkdir -p "$(dirname "$SEMIKB_DATA_ROOT")"
tar -C "$(dirname "$SEMIKB_DATA_ROOT")" -xzf "$archive"
install -m 0600 "$env_backup" .env
echo "Cold restore completed. Run host_preflight.sh, then start the stack and verify resources."
