#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

seed_demo=false
if [[ "${1:-}" == "--seed-demo" ]]; then
  seed_demo=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--seed-demo]" >&2
  exit 2
fi

./scripts/deployment/host_preflight.sh .env
./scripts/deployment/prepare_data_dirs.sh .env
docker compose --env-file .env -f docker-compose.prod.yml config --quiet
docker compose --env-file .env -f docker-compose.prod.yml pull --policy missing \
  etcd milvus-minio milvus mongodb minio redis
docker compose --env-file .env -f docker-compose.prod.yml build api worker web milvus-init
docker compose --env-file .env -f docker-compose.prod.yml up -d etcd milvus-minio milvus mongodb minio redis
docker compose --env-file .env -f docker-compose.prod.yml up -d milvus-init
milvus_init_id="$(docker compose --env-file .env -f docker-compose.prod.yml ps -q milvus-init)"
[[ -n "$milvus_init_id" ]] || { echo "Milvus initialization container was not created." >&2; exit 1; }
milvus_init_status="$(docker wait "$milvus_init_id")"
if [[ "$milvus_init_status" != "0" ]]; then
  docker compose --env-file .env -f docker-compose.prod.yml logs --no-color milvus-init
  echo "Milvus authentication initialization failed." >&2
  exit 1
fi
docker compose --env-file .env -f docker-compose.prod.yml run --rm --no-deps api python -m semikb.storage.provisioning
if [[ "$seed_demo" == "true" ]]; then
  docker compose --env-file .env -f docker-compose.prod.yml run --rm --no-deps api \
    python -m semikb.deployment.seed_demo_corpus --apply
fi
docker compose --env-file .env -f docker-compose.prod.yml up -d api worker
# Nginx resolves the API container when it starts. Recreate Web after an API image update
# so an old upstream container IP cannot leave the public route returning 502.
docker compose --env-file .env -f docker-compose.prod.yml up -d --force-recreate web
docker compose --env-file .env -f docker-compose.prod.yml ps
if [[ "$seed_demo" == "true" ]]; then
  echo "Services started with the governed CC0 synthetic corpus."
else
  echo "Services started. Provisioning did not publish the active Milvus alias or seed knowledge."
fi
