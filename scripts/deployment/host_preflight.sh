#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/.env}"
MIN_MEMORY_KIB=$((7 * 1024 * 1024))
MIN_DISK_KIB=$((55 * 1024 * 1024))
REQUIRED_KEYS=(
  SEMIKB_DATA_ROOT SEMIKB_BACKUP_ROOT JWT_SECRET DEMO_ACCESS_KEY
  MONGO_ROOT_USERNAME MONGO_ROOT_PASSWORD MONGO_APP_USERNAME MONGO_APP_PASSWORD
  MILVUS_ROOT_PASSWORD MILVUS_MINIO_ROOT_USER MILVUS_MINIO_ROOT_PASSWORD
  MINIO_ROOT_USER MINIO_ROOT_PASSWORD REDIS_PASSWORD
  MINERU_API_KEY CLOSEAI_API_KEY QWEN_API_KEY RERANK_API_KEY ALIYUN_WEB_MCP_API_KEY
)

failures=0
pass() { printf '[PASS] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1" >&2; failures=$((failures + 1)); }

[[ -f "$ENV_FILE" ]] || { fail "environment file is missing: $ENV_FILE"; exit 1; }
permissions="$(stat -c '%a' "$ENV_FILE")"
[[ "$permissions" == "600" ]] && pass ".env permissions are 600" || fail ".env permissions must be 600 (current: $permissions)"

env_value() {
  awk -v key="$1" 'index($0, key "=") == 1 {sub(/^[^=]*=/, ""); sub(/\r$/, ""); print; exit}' "$ENV_FILE"
}

for key in "${REQUIRED_KEYS[@]}"; do
  [[ -n "$(env_value "$key")" ]] || fail "required setting is missing: $key"
done
for key in JWT_SECRET DEMO_ACCESS_KEY MONGO_ROOT_PASSWORD MONGO_APP_PASSWORD MILVUS_ROOT_PASSWORD MILVUS_MINIO_ROOT_PASSWORD MINIO_ROOT_PASSWORD REDIS_PASSWORD; do
  value="$(env_value "$key")"
  [[ ${#value} -ge 16 && "$value" != *replace-with* ]] || fail "$key must be a generated secret of at least 16 characters"
done
[[ "$(env_value APP_ENV)" == "production" ]] && pass "APP_ENV is production" || fail "APP_ENV must be production"
[[ "$(env_value DEMO_MODE)" == "false" ]] && pass "DEMO_MODE is false" || fail "DEMO_MODE must be false"

memory_kib="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
[[ "$memory_kib" -ge "$MIN_MEMORY_KIB" ]] && pass "memory is at least 7 GiB" || fail "memory is below the 8 GiB deployment class"
cpu_count="$(getconf _NPROCESSORS_ONLN)"
[[ "$cpu_count" -ge 4 ]] && pass "CPU count is at least 4" || fail "at least 4 vCPU are required"
grep -qE '\bavx2?\b' /proc/cpuinfo && pass "CPU exposes AVX/AVX2" || fail "Milvus requires AVX-capable CPU"

data_root="$(env_value SEMIKB_DATA_ROOT)"
data_parent="$(dirname "${data_root:-/opt/semiconductor-agent-knowledge-base-data}")"
disk_kib="$(df -Pk "$data_parent" | awk 'NR==2 {print $4}')"
[[ "$disk_kib" -ge "$MIN_DISK_KIB" ]] && pass "free disk is at least 55 GiB" || fail "free disk is below the deployment reserve"

command -v docker >/dev/null && pass "Docker CLI is installed" || fail "Docker CLI is missing"
docker info >/dev/null 2>&1 && pass "Docker daemon is reachable" || fail "Docker daemon is not reachable"
docker compose version >/dev/null 2>&1 && pass "Docker Compose plugin is installed" || fail "Docker Compose plugin is missing"

if command -v ss >/dev/null && ss -ltnH | awk '{print $4}' | grep -qE '(^|:)80$'; then
  fail "TCP port 80 is already in use"
else
  pass "TCP port 80 is available"
fi

if [[ "$failures" -gt 0 ]]; then
  printf 'Preflight failed with %d issue(s). No deployment action was performed.\n' "$failures" >&2
  exit 1
fi
printf 'Preflight passed. Security-group rules must still be verified in Alibaba Cloud.\n'
