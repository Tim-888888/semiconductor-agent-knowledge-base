#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "${1:-}" != "--apply" || "${T946_RESTART_CONFIRM:-}" != "restart-t946-services" ]]; then
  echo "Refusing restart. Use --apply with T946_RESTART_CONFIRM=restart-t946-services." >&2
  exit 2
fi

OUTPUT_DIR="${2:-$ROOT_DIR/docs/evidence/t9-4-6/restarts}"
mkdir -p "$OUTPUT_DIR"
COMPOSE=(docker compose --env-file .env -f docker-compose.prod.yml)
SERVICES=(web api worker redis minio mongodb milvus etcd milvus-minio)

compose_id() {
  "${COMPOSE[@]}" ps -q "$1"
}

snapshot() {
  local destination="$1"
  : > "$destination"
  local service id
  for service in "${SERVICES[@]}"; do
    id="$(compose_id "$service")"
    [[ -n "$id" ]] || { echo "service is missing: $service" >&2; exit 1; }
    docker inspect "$id" | python3 -c '
import json
import sys

item = json.load(sys.stdin)[0]
state = item["State"]
health = state.get("Health", {}).get("Status", "none")
print(
    "{}|{}|{}|{}|{}|{}".format(
        item["Id"],
        item["Name"],
        state["Status"],
        health,
        item["RestartCount"],
        str(state["OOMKilled"]).lower(),
    )
)
' >> "$destination"
  done
}

wait_running() {
  local service="$1" deadline=$((SECONDS + 180)) id status health
  while ((SECONDS < deadline)); do
    id="$(compose_id "$service")"
    if [[ -n "$id" ]]; then
      status="$(docker inspect --format '{{.State.Status}}' "$id")"
      health="$(
        docker inspect "$id" \
          | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["State"].get("Health", {}).get("Status", "none"))'
      )"
      if [[ "$status" == "running" && ("$health" == "healthy" || "$health" == "none") ]]; then
        return 0
      fi
    fi
    sleep 2
  done
  echo "service did not recover: $service" >&2
  return 1
}

worker_ping() {
  local deadline=$((SECONDS + 90))
  while ((SECONDS < deadline)); do
    if "${COMPOSE[@]}" exec -T worker \
      celery -A semikb.workers.celery_app:celery_app inspect ping --timeout 10 2>/dev/null \
      | grep -q pong; then
      return 0
    fi
    sleep 3
  done
  echo "Celery worker did not recover" >&2
  return 1
}

snapshot "$OUTPUT_DIR/before.txt"
curl -fsS http://127.0.0.1/healthz >/dev/null
curl -fsS http://127.0.0.1/api/v1/live >/dev/null
worker_ping

for service in "${SERVICES[@]}"; do
  printf 'restarting %s\n' "$service" | tee -a "$OUTPUT_DIR/sequence.txt"
  "${COMPOSE[@]}" restart "$service" >/dev/null
  wait_running "$service"
  if [[ "$service" == "api" || "$service" == "web" ]]; then
    curl -fsS --retry 20 --retry-delay 2 http://127.0.0.1/healthz >/dev/null
    curl -fsS --retry 20 --retry-delay 2 http://127.0.0.1/api/v1/live >/dev/null
  fi
  if [[ "$service" == "worker" || "$service" == "redis" ]]; then
    worker_ping
  fi
done

curl -fsS --retry 20 --retry-delay 2 http://127.0.0.1/healthz >/dev/null
curl -fsS --retry 20 --retry-delay 2 http://127.0.0.1/api/v1/live >/dev/null
worker_ping
snapshot "$OUTPUT_DIR/after.txt"

before_ids="$(cut -d'|' -f1 "$OUTPUT_DIR/before.txt")"
after_ids="$(cut -d'|' -f1 "$OUTPUT_DIR/after.txt")"
[[ "$before_ids" == "$after_ids" ]] || {
  echo "container identities changed during restart verification" >&2
  exit 1
}
if grep -q '|true$' "$OUTPUT_DIR/after.txt"; then
  echo "an OOM-killed container was detected" >&2
  exit 1
fi
echo "T9-4.6 controlled restart sequence passed."
