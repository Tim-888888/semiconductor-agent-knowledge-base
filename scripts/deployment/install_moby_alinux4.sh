#!/usr/bin/env bash
set -euo pipefail

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "Docker-compatible engine and Compose plugin already exist; no package changes made."
  exit 0
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "alinux" || "${VERSION_ID:-}" != 4* ]]; then
  echo "This installer only supports Alibaba Cloud Linux 4." >&2
  exit 1
fi

dnf install -y moby docker-compose-plugin
systemctl enable --now docker
docker version
docker compose version
echo "Moby and the Docker Compose plugin are ready."
