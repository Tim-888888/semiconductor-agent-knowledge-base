#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  echo "Usage: T949_ACCEPTANCE_CONFIRM=run-t949-final $0 --apply --output-dir <dir> --bundle-dir <dir>" >&2
  exit 2
}

apply=false
output_dir=""
bundle_dir=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      apply=true
      ;;
    --output-dir)
      shift
      [[ $# -gt 0 ]] || usage
      output_dir="$1"
      ;;
    --bundle-dir)
      shift
      [[ $# -gt 0 ]] || usage
      bundle_dir="$1"
      ;;
    *)
      usage
      ;;
  esac
  shift
done

[[ "$apply" == "true" ]] || usage
[[ "${T949_ACCEPTANCE_CONFIRM:-}" == "run-t949-final" ]] || {
  echo "Set T949_ACCEPTANCE_CONFIRM=run-t949-final." >&2
  exit 1
}
[[ -n "$output_dir" && -n "$bundle_dir" ]] || usage
[[ -f .env ]] || { echo "Missing root .env." >&2; exit 1; }
[[ -d "$bundle_dir" ]] || { echo "Offline bundle does not exist: $bundle_dir" >&2; exit 1; }
if [[ -e "$output_dir" ]] && [[ -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty evidence directory: $output_dir" >&2
  exit 1
fi
mkdir -p "$output_dir"

compose=(docker compose --env-file .env -f docker-compose.prod.yml)
"${compose[@]}" config --quiet
"${compose[@]}" ps > "$output_dir/compose-ps.txt"

docker exec semikb-api-1 python scripts/verify_t949_final_demo.py \
  --base-url http://web > "$output_dir/final-demo.json"

python3 scripts/verify_t948_security.py \
  --env .env --base-url http://127.0.0.1 \
  --test-rate-limit --wait-asset-expiry \
  --output "$output_dir/security.json"

python3 scripts/verify_t948_offline_bundle.py \
  --bundle-dir "$bundle_dir" --verify-docker \
  --output "$output_dir/offline.json"

docker exec semikb-api-1 python scripts/verify_t947_restore.py \
  --retrieval-smoke > "$output_dir/storage-state.json"
docker exec semikb-worker-1 celery -A semikb.workers.celery_app inspect ping --timeout 10 \
  > "$output_dir/worker-ping.txt"
curl -fsS http://127.0.0.1/api/v1/health > "$output_dir/health.json"
ss -lntup > "$output_dir/listeners.txt"

python3 scripts/summarize_t949_acceptance.py \
  --evidence-dir "$output_dir" \
  --baseline docs/evidence/t9-4-8/state-after.json \
  --output "$output_dir/final-verdict.json"

echo "T9-4.9 automated acceptance passed: $output_dir"
