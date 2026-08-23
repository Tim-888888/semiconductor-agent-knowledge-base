#!/usr/bin/env bash
set -euo pipefail

restore_env=""
project_name=""
confirm_copy=0
apply=0
while (($#)); do
  case "$1" in
    --restore-env) restore_env="${2:-}"; shift 2 ;;
    --project-name) project_name="${2:-}"; shift 2 ;;
    --confirm-restore-copy) confirm_copy=1; shift ;;
    --apply) apply=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$restore_env" || -z "$project_name" || "$confirm_copy" != 1 || "$apply" != 1 \
  || "${T947_CLEANUP_CONFIRM:-}" != "remove-t947-restore-copy" ]]; then
  echo "Refusing cleanup. Provide --restore-env, --project-name, --confirm-restore-copy, --apply and T947_CLEANUP_CONFIRM=remove-t947-restore-copy." >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
[[ "$restore_env" = /* && -f "$restore_env" ]] || {
  echo "Restore environment must be an existing absolute file." >&2
  exit 2
}
[[ "$project_name" =~ ^[a-z0-9][a-z0-9_-]{2,48}$ && "$project_name" != "semikb" ]] || {
  echo "Refusing cleanup for the live or invalid Compose project name." >&2
  exit 2
}

env_value() {
  awk -v key="$2" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); sub(/\r$/, ""); print; exit}' "$1"
}
target_data_root="$(realpath -m -- "$(env_value "$restore_env" SEMIKB_DATA_ROOT)")"
live_data_root="$(realpath -m -- "$(env_value "$ROOT_DIR/.env" SEMIKB_DATA_ROOT)")"
env_project="$(env_value "$restore_env" COMPOSE_PROJECT_NAME)"
[[ "$target_data_root" != "/" && "$target_data_root" != "$live_data_root" ]] || {
  echo "Refusing cleanup because target matches the live or root data directory." >&2
  exit 2
}
[[ "$env_project" == "$project_name" ]] || {
  echo "Restore environment project does not match the requested project." >&2
  exit 2
}
marker="$target_data_root/.t947-restore-marker.json"
[[ -f "$marker" ]] || { echo "Restore marker is missing: $marker" >&2; exit 1; }
python3 - "$marker" "$target_data_root" "$project_name" <<'PY'
import json
import sys

marker, target, project = sys.argv[1:]
payload = json.load(open(marker, encoding="utf-8"))
if payload.get("target_data_root") != target or payload.get("project_name") != project:
    raise SystemExit("Restore marker does not match the requested cleanup target.")
PY

COMPOSE=(docker compose -p "$project_name" --env-file "$restore_env" -f docker-compose.prod.yml)
"${COMPOSE[@]}" down --remove-orphans
if docker ps -aq --filter "label=com.docker.compose.project=$project_name" | grep -q .; then
  echo "Refusing data cleanup because restore project containers still exist." >&2
  exit 1
fi
rm -rf -- "$target_data_root"
rm -f -- "$restore_env"
echo "Independent T9-4.7 restore copy removed: $target_data_root"
