#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m uvicorn semikb.api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill "$API_PID"' EXIT

cd web
npm run dev -- --host 127.0.0.1 --port 5173
