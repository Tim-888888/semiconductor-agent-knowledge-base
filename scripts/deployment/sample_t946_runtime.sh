#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/docs/evidence/t9-4-6/runtime}"
SAMPLES="${2:-120}"
INTERVAL_SECONDS="${3:-2}"

[[ "$SAMPLES" =~ ^[0-9]+$ ]] && ((SAMPLES >= 1 && SAMPLES <= 900)) \
  || { echo "samples must be within 1..900" >&2; exit 2; }
[[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] && ((INTERVAL_SECONDS >= 1 && INTERVAL_SECONDS <= 60)) \
  || { echo "interval must be within 1..60 seconds" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
: > "$OUTPUT_DIR/host.jsonl"
: > "$OUTPUT_DIR/containers.jsonl"

snapshot_state() {
  local destination="$1"
  : > "$destination"
  local container
  while IFS= read -r container; do
    [[ -n "$container" ]] || continue
    docker inspect --format \
      '{{.Id}}|{{.Name}}|{{.State.Status}}|{{.RestartCount}}|{{.State.OOMKilled}}|{{.LogPath}}' \
      "$container" >> "$destination"
  done < <(docker ps --format '{{.Names}}' | sort)
}

snapshot_state "$OUTPUT_DIR/container-state-before.txt"

stop_requested=0
trap 'stop_requested=1' TERM INT

for ((index=1; index<=SAMPLES; index++)); do
  timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  mem_available_kib="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
  swap_total_kib="$(awk '/SwapTotal/ {print $2}' /proc/meminfo)"
  swap_free_kib="$(awk '/SwapFree/ {print $2}' /proc/meminfo)"
  root_available_bytes="$(df -B1 / | awk 'NR==2 {print $4}')"
  load_1m="$(awk '{print $1}' /proc/loadavg)"
  printf '{"timestamp":"%s","mem_available_kib":%s,"swap_total_kib":%s,"swap_free_kib":%s,"root_available_bytes":%s,"load_1m":%s}\n' \
    "$timestamp" "$mem_available_kib" "$swap_total_kib" "$swap_free_kib" \
    "$root_available_bytes" "$load_1m" >> "$OUTPUT_DIR/host.jsonl"
  docker stats --no-stream --format '{{json .}}' >> "$OUTPUT_DIR/containers.jsonl"
  if ((stop_requested == 1)); then
    break
  fi
  if ((index < SAMPLES)); then
    sleep "$INTERVAL_SECONDS" || true
    if ((stop_requested == 1)); then
      break
    fi
  fi
done

snapshot_state "$OUTPUT_DIR/container-state-after.txt"
echo "T9-4.6 runtime samples written to $OUTPUT_DIR"
