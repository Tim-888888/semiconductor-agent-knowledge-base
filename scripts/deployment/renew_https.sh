#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

env_file="${SEMIKB_APP_ENV_FILE:-.env}"

[[ -f "$env_file" ]] || { echo "Missing environment file: $env_file" >&2; exit 1; }

read_env_value() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key { value = substr($0, index($0, "=") + 1) } END { print value }' "$env_file"
}

certbot_root="${SEMIKB_CERTBOT_ROOT:-$(read_env_value SEMIKB_CERTBOT_ROOT)}"
certbot_image="${SEMIKB_CERTBOT_IMAGE:-$(read_env_value SEMIKB_CERTBOT_IMAGE)}"
certbot_root="${certbot_root:-/opt/semiconductor-agent-knowledge-base-certbot}"
certbot_image="${certbot_image:-certbot/certbot:v5.6.0}"
[[ -d "$certbot_root/conf/live" ]] || { echo "No managed certificate exists under $certbot_root/conf." >&2; exit 1; }

docker run --rm \
  --volume "$certbot_root/www:/var/www/certbot" \
  --volume "$certbot_root/conf:/etc/letsencrypt" \
  "$certbot_image" renew --webroot --webroot-path /var/www/certbot --non-interactive

find "$certbot_root/conf/archive" -type f -name 'privkey*.pem' -exec chmod 0640 {} +
docker compose --env-file "$env_file" \
  -f docker-compose.prod.yml -f docker-compose.https.yml kill --signal HUP web
