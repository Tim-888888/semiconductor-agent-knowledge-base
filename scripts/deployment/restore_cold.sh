#!/usr/bin/env bash
set -euo pipefail

archive=""
env_backup=""
target_data_root=""
restore_env=""
project_name=""
confirm_empty=0
apply=0
while (($#)); do
  case "$1" in
    --archive) archive="${2:-}"; shift 2 ;;
    --env) env_backup="${2:-}"; shift 2 ;;
    --target-data-root) target_data_root="${2:-}"; shift 2 ;;
    --restore-env) restore_env="${2:-}"; shift 2 ;;
    --project-name) project_name="${2:-}"; shift 2 ;;
    --confirm-empty-target) confirm_empty=1; shift ;;
    --apply) apply=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$archive" || -z "$env_backup" || -z "$target_data_root" || -z "$restore_env" \
  || -z "$project_name" || "$confirm_empty" != 1 || "$apply" != 1 ]]; then
  echo "Usage: $0 --archive /absolute/backup.tar.gz --env /absolute/backup.env --target-data-root /absolute/empty-dir --restore-env /absolute/restore.env --project-name isolated-name --confirm-empty-target --apply" >&2
  exit 2
fi
if [[ "${T947_RESTORE_CONFIRM:-}" != "restore-t947-independent" ]]; then
  echo "Refusing restore. Set T947_RESTORE_CONFIRM=restore-t947-independent." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
[[ "$archive" = /* && "$env_backup" = /* && "$target_data_root" = /* && "$restore_env" = /* ]] || {
  echo "Archive, environment, target, and restore environment paths must be absolute." >&2
  exit 2
}
[[ "$project_name" =~ ^[a-z0-9][a-z0-9_-]{2,48}$ && "$project_name" != "semikb" ]] || {
  echo "Project name must be a non-live lowercase Compose project name." >&2
  exit 2
}

archive="$(realpath -m -- "$archive")"
env_backup="$(realpath -m -- "$env_backup")"
target_data_root="$(realpath -m -- "$target_data_root")"
restore_env="$(realpath -m -- "$restore_env")"
[[ -f "$env_backup" && -f "$env_backup.sha256" ]] || { echo "Environment backup or checksum is missing." >&2; exit 1; }
expected_env_hash="$(awk 'NR==1 {print $1}' "$env_backup.sha256")"
actual_env_hash="$(sha256sum "$env_backup" | awk '{print $1}')"
[[ "$actual_env_hash" == "$expected_env_hash" ]] || { echo "Environment checksum validation failed." >&2; exit 1; }
env_value() {
  awk -v key="$2" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); sub(/\r$/, ""); print; exit}' "$1"
}
source_data_root_value="$(env_value "$env_backup" SEMIKB_DATA_ROOT)"
: "${source_data_root_value:?SEMIKB_DATA_ROOT is required in the backup environment}"
source_data_root="$(realpath -m -- "$source_data_root_value")"
[[ "$source_data_root" != "/" && "$target_data_root" != "/" ]] || {
  echo "Source and target data roots must not be filesystem root." >&2
  exit 2
}
[[ "$target_data_root" != "$source_data_root" ]] || {
  echo "Restore target must differ from the source data root." >&2
  exit 2
}
case "$target_data_root/" in
  "$source_data_root/"*) echo "Restore target cannot be inside the source data root." >&2; exit 2 ;;
esac
case "$source_data_root/" in
  "$target_data_root/"*) echo "Source data root cannot be inside the restore target." >&2; exit 2 ;;
esac
case "$restore_env/" in
  "$target_data_root/"*) echo "Restore environment must be outside the restored data root." >&2; exit 2 ;;
esac
[[ "$restore_env" != "$ROOT_DIR/.env" && "$restore_env" != "$env_backup" ]] || {
  echo "Restore environment must not overwrite the live or backup environment file." >&2
  exit 2
}
[[ -f "$archive" && -f "$archive.sha256" ]] || { echo "Archive or checksum is missing." >&2; exit 1; }
expected_hash="$(awk 'NR==1 {print $1}' "$archive.sha256")"
actual_hash="$(sha256sum "$archive" | awk '{print $1}')"
[[ "$actual_hash" == "$expected_hash" ]] || { echo "Archive checksum validation failed." >&2; exit 1; }

if [[ -e "$restore_env" ]]; then
  echo "Refusing restore because restore environment already exists: $restore_env" >&2
  exit 1
fi
if [[ -d "$target_data_root" ]] && find "$target_data_root" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing restore because target is not empty: $target_data_root" >&2
  exit 1
fi
[[ ! -e "$target_data_root" || -d "$target_data_root" ]] || {
  echo "Restore target exists but is not a directory: $target_data_root" >&2
  exit 1
}

source_basename="$(basename "$source_data_root")"
python3 - "$archive" "$source_basename" <<'PY'
from __future__ import annotations

import sys
import tarfile
from pathlib import PurePosixPath

archive, expected_root = sys.argv[1:]
with tarfile.open(archive, "r:gz") as handle:
    members = handle.getmembers()
    if not members:
        raise SystemExit("Backup archive is empty.")
    for member in members:
        path = PurePosixPath(member.name)
        parts = tuple(part for part in path.parts if part not in ("", "."))
        if path.is_absolute() or ".." in parts or not parts or parts[0] != expected_root:
            raise SystemExit(f"Unsafe archive member: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"Unsupported archive member type: {member.name}")
PY

target_parent="$(dirname "$target_data_root")"
install -d -m 0750 "$target_parent" "$(dirname "$restore_env")"
staging="$(mktemp -d "$target_parent/.t947-restore-staging.XXXXXX")"
cleanup_staging() {
  rm -rf -- "$staging"
}
trap cleanup_staging EXIT
tar --numeric-owner -C "$staging" -xzf "$archive"
[[ -d "$staging/$source_basename" ]] || {
  echo "Expected archive root is missing after extraction: $source_basename" >&2
  exit 1
}
if [[ -d "$target_data_root" ]]; then
  rmdir -- "$target_data_root"
fi
mv -- "$staging/$source_basename" "$target_data_root"

python3 - "$env_backup" "$restore_env" "$target_data_root" "$project_name" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

source, destination, target, project = map(Path, sys.argv[1:])
updates = {
    "SEMIKB_DATA_ROOT": str(target),
    "SEMIKB_BACKUP_ROOT": f"{target}-backups",
    "SEMIKB_APP_ENV_FILE": str(destination),
    "COMPOSE_PROJECT_NAME": str(project),
}
seen: set[str] = set()
output: list[str] = []
for raw in source.read_text(encoding="utf-8").splitlines():
    key = raw.split("=", 1)[0] if "=" in raw and not raw.lstrip().startswith("#") else ""
    if key in updates:
        output.append(f"{key}={updates[key]}")
        seen.add(key)
    else:
        output.append(raw)
for key, value in updates.items():
    if key not in seen:
        output.append(f"{key}={value}")
destination.write_text("\n".join(output) + "\n", encoding="utf-8")
os.chmod(destination, 0o600)
PY

marker="$target_data_root/.t947-restore-marker.json"
plan="$restore_env.plan.json"
python3 - "$marker" "$plan" "$archive" "$actual_hash" "$source_data_root" \
  "$target_data_root" "$restore_env" "$project_name" <<'PY'
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

marker, plan, archive, archive_hash, source, target, restore_env, project = sys.argv[1:]
payload = {
    "schema": "semikb-t947-independent-restore-v1",
    "prepared_at": datetime.now(timezone.utc).isoformat(),
    "archive_name": Path(archive).name,
    "archive_sha256": archive_hash,
    "source_data_root": source,
    "target_data_root": target,
    "restore_env": restore_env,
    "project_name": project,
    "live_environment_overwritten": False,
}
for path in (Path(marker), Path(plan)):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
PY

trap - EXIT
cleanup_staging
echo "Cold restore prepared in independent data root: $target_data_root"
echo "Protected restore environment created: $restore_env"
echo "Restore plan created: $plan"
echo "Start only the isolated project with docker compose -p $project_name --env-file $restore_env."
