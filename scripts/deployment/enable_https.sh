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
primary_domain="${SEMIKB_PRIMARY_DOMAIN:-$(read_env_value SEMIKB_PRIMARY_DOMAIN)}"
www_domain="${SEMIKB_WWW_DOMAIN:-$(read_env_value SEMIKB_WWW_DOMAIN)}"
certbot_image="${SEMIKB_CERTBOT_IMAGE:-$(read_env_value SEMIKB_CERTBOT_IMAGE)}"
certbot_email="${SEMIKB_CERTBOT_EMAIL:-$(read_env_value SEMIKB_CERTBOT_EMAIL)}"
certbot_root="${certbot_root:-/opt/semiconductor-agent-knowledge-base-certbot}"
primary_domain="${primary_domain:-semiatlas.cn}"
www_domain="${www_domain:-www.semiatlas.cn}"
certbot_image="${certbot_image:-certbot/certbot:v5.6.0}"
[[ "$primary_domain" == "semiatlas.cn" ]] || {
  echo "The checked-in Nginx TLS configuration currently supports semiatlas.cn only." >&2
  exit 1
}
[[ "$www_domain" == "www.semiatlas.cn" ]] || {
  echo "The checked-in Nginx TLS configuration currently supports www.semiatlas.cn only." >&2
  exit 1
}

install -d -m 0755 "$certbot_root/www/.well-known/acme-challenge" "$certbot_root/conf"

compose=(docker compose --env-file "$env_file" -f docker-compose.prod.yml)
acme_compose=("${compose[@]}" -f docker-compose.acme.yml)
https_compose=("${compose[@]}" -f docker-compose.https.yml)

"${acme_compose[@]}" config --quiet
"${acme_compose[@]}" up -d --no-deps --force-recreate web

probe="semikb-acme-$(date +%s)"
printf '%s\n' "$probe" >"$certbot_root/www/.well-known/acme-challenge/$probe"
trap 'rm -f "$certbot_root/www/.well-known/acme-challenge/$probe"' EXIT
curl --fail --silent --show-error --max-time 15 \
  "http://127.0.0.1/.well-known/acme-challenge/$probe" | grep -Fx "$probe" >/dev/null

certbot_args=(
  certonly
  --webroot
  --webroot-path /var/www/certbot
  --domain "$primary_domain"
  --domain "$www_domain"
  --agree-tos
  --non-interactive
  --no-eff-email
  --keep-until-expiring
)
if [[ -n "$certbot_email" ]]; then
  certbot_args+=(--email "$certbot_email")
else
  certbot_args+=(--register-unsafely-without-email)
fi

docker pull "$certbot_image"
docker run --rm \
  --volume "$certbot_root/www:/var/www/certbot" \
  --volume "$certbot_root/conf:/etc/letsencrypt" \
  "$certbot_image" "${certbot_args[@]}"

find "$certbot_root/conf/archive" -type f -name 'privkey*.pem' -exec chmod 0640 {} +
"${https_compose[@]}" config --quiet
"${https_compose[@]}" up -d --no-deps --force-recreate web
curl --fail --silent --show-error --max-time 15 \
  --resolve "$primary_domain:443:127.0.0.1" "https://$primary_domain/healthz" >/dev/null

install -m 0644 deploy/systemd/semikb-certbot-renew.service /etc/systemd/system/
install -m 0644 deploy/systemd/semikb-certbot-renew.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now semikb-certbot-renew.timer

echo "HTTPS enabled for $primary_domain and $www_domain."
