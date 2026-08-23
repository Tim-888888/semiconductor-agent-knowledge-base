"""Run credential-safe T9-4.8 security checks on the production host."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error, parse, request

APP_SERVICES = ("api", "worker", "web")
EXPECTED_PRODUCTION_SCOPE = {
    "user_id": "demo_engineer",
    "roles": ["engineer", "knowledge_admin"],
    "access_scope_keys": ["demo_engineering"],
    "fabs": ["FAB-01"],
    "products": ["P-ALPHA"],
    "tool_ids": ["ETCH-03"],
}
PUBLIC_ALLOWED_PORTS = {22, 80}
DATA_PORTS = {2379, 27017, 6379, 9000, 9001, 9091, 19530}
SECRET_NAME_MARKERS = ("_API_KEY", "_ACCESS_KEY", "_PASSWORD", "_SECRET", "_TOKEN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--base-url", default="http://127.0.0.1")
    parser.add_argument("--project-name", default="semikb")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--test-rate-limit", action="store_true")
    parser.add_argument("--wait-asset-expiry", action="store_true")
    return parser.parse_args()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def encode_hs256(payload: dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    segments = [
        b64url(json.dumps(item, separators=(",", ":")).encode())
        for item in (header, payload)
    ]
    signing_input = ".".join(segments).encode()
    segments.append(b64url(hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()))
    return ".".join(segments)


def decode_payload(token: str) -> dict[str, Any]:
    segment = token.split(".")[1]
    segment += "=" * ((4 - len(segment) % 4) % 4)
    return json.loads(base64.urlsafe_b64decode(segment))


def http_call(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    timeout: float = 20,
) -> tuple[int, bytes, dict[str, str]]:
    data = None if payload is None else json.dumps(payload).encode()
    actual_headers = dict(headers or {})
    if payload is not None:
        actual_headers["Content-Type"] = "application/json"
    req = request.Request(url, data=data, headers=actual_headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read(), dict(response.headers.items())
    except error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers.items())


def run(*command: str, cwd: Path | None = None, timeout: int = 120) -> bytes:
    return subprocess.check_output(command, cwd=cwd, timeout=timeout, stderr=subprocess.STDOUT)


def compose_containers(project_name: str) -> dict[str, dict[str, Any]]:
    names = run(
        "docker",
        "ps",
        "--filter",
        f"label=com.docker.compose.project={project_name}",
        "--format",
        "{{.Names}}",
    ).decode().splitlines()
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        inspected = json.loads(run("docker", "inspect", name))[0]
        service = inspected["Config"]["Labels"]["com.docker.compose.service"]
        result[service] = inspected
    return result


def is_root_user(configured_user: str) -> bool:
    normalized = configured_user.strip().lower()
    return normalized in {"", "0", "0:0", "root", "root:root"}


def public_listeners() -> list[int]:
    output = run("ss", "-H", "-lnt").decode("utf-8", "replace")
    listeners: set[int] = set()
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        local = fields[3]
        if not (local.startswith("0.0.0.0:") or local.startswith("[::]:") or local.startswith("*:") or local.startswith(":::")):
            continue
        match = re.search(r":(\d+)$", local)
        if match:
            listeners.add(int(match.group(1)))
    return sorted(listeners)


def tracked_file_bytes(root: Path) -> bytes:
    paths = run("git", "ls-files", "-z", cwd=root).split(b"\0")
    chunks: list[bytes] = []
    for raw_path in paths:
        if not raw_path:
            continue
        path = root / os.fsdecode(raw_path)
        if path.is_file():
            chunks.append(path.read_bytes())
    return b"\n".join(chunks)


def secret_hits(values: dict[str, str], surfaces: dict[str, bytes]) -> list[dict[str, str]]:
    secrets = {
        key: value.encode()
        for key, value in values.items()
        if len(value) >= 12 and any(marker in key for marker in SECRET_NAME_MARKERS)
    }
    hits: list[dict[str, str]] = []
    for surface, content in surfaces.items():
        for key, value in secrets.items():
            if value in content:
                hits.append({"surface": surface, "secret_key": key})
    return hits


def find_accessible_image_id(
    containers: dict[str, dict[str, Any]],
    base_url: str,
    token: str,
) -> tuple[str | None, dict[str, Any] | None]:
    mongodb = containers.get("mongodb")
    if not mongodb:
        return None, None
    shell = (
        'mongosh --quiet --username "$MONGO_INITDB_ROOT_USERNAME" '
        '--password "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin semikb '
        "--eval 'print(JSON.stringify(db.image_assets.find({}, {_id:0,image_id:1}).limit(100).toArray()))'"
    )
    raw = run("docker", "exec", mongodb["Name"].lstrip("/"), "sh", "-lc", shell)
    candidates = json.loads(raw.decode().strip() or "[]")
    headers = {"Authorization": f"Bearer {token}"}
    for candidate in candidates:
        image_id = candidate.get("image_id")
        if not image_id:
            continue
        status, body, _ = http_call(
            f"{base_url}/api/v1/assets/{parse.quote(image_id, safe='')}/access",
            headers=headers,
        )
        if status == 200:
            return image_id, json.loads(body)
    return None, None


def token_for_scope(scope: dict[str, Any], values: dict[str, str], *, expired: bool = False) -> str:
    point = datetime.now(UTC) - timedelta(minutes=5) if expired else datetime.now(UTC) + timedelta(minutes=10)
    return encode_hs256(
        {"sub": scope["user_id"], "scope": scope, "exp": int(point.timestamp())},
        values["JWT_SECRET"],
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    values = read_env(args.env.resolve())
    base_url = args.base_url.rstrip("/")
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    containers = compose_containers(args.project_name)
    listeners = public_listeners()
    check("public_listener_allowlist", set(listeners) <= PUBLIC_ALLOWED_PORTS, listeners)
    check("data_ports_not_public", not DATA_PORTS.intersection(listeners), listeners)

    app_runtime: dict[str, Any] = {}
    for service in APP_SERVICES:
        inspected = containers.get(service)
        if not inspected:
            app_runtime[service] = {"present": False}
            continue
        host = inspected["HostConfig"]
        user = inspected["Config"].get("User") or ""
        app_runtime[service] = {
            "present": True,
            "configured_user": user or "root-default",
            "non_root": not is_root_user(user),
            "read_only": bool(host.get("ReadonlyRootfs")),
            "cap_drop_all": "ALL" in (host.get("CapDrop") or []),
            "no_new_privileges": "no-new-privileges:true" in (host.get("SecurityOpt") or []),
            "privileged": bool(host.get("Privileged")),
        }
    check(
        "application_container_hardening",
        all(
            item.get("present")
            and item.get("non_root")
            and item.get("read_only")
            and item.get("cap_drop_all")
            and item.get("no_new_privileges")
            and not item.get("privileged")
            for item in app_runtime.values()
        ),
        app_runtime,
    )
    env_mode = oct(args.env.stat().st_mode & 0o777)[2:]
    check("environment_file_mode", env_mode == "600", env_mode)

    missing_status, _, _ = http_call(f"{base_url}/api/v1/auth/demo-token", method="POST")
    wrong_status, _, _ = http_call(
        f"{base_url}/api/v1/auth/demo-token",
        method="POST",
        headers={"X-Demo-Access-Key": "deliberately-invalid-access-key"},
    )
    correct_status, correct_body, _ = http_call(
        f"{base_url}/api/v1/auth/demo-token",
        method="POST",
        headers={"X-Demo-Access-Key": values["DEMO_ACCESS_KEY"]},
        payload={
            "user_id": "client_selected_identity",
            "roles": ["admin"],
            "access_scope_keys": ["client_selected_scope"],
        },
    )
    token = json.loads(correct_body).get("access_token", "") if correct_status == 200 else ""
    claims = decode_payload(token) if token else {}
    check(
        "demo_access_key_and_server_owned_scope",
        missing_status == 401
        and wrong_status == 401
        and correct_status == 200
        and claims.get("sub") == EXPECTED_PRODUCTION_SCOPE["user_id"]
        and claims.get("scope") == EXPECTED_PRODUCTION_SCOPE,
        {
            "missing_status": missing_status,
            "wrong_status": wrong_status,
            "correct_status": correct_status,
            "issued_sub": claims.get("sub"),
            "issued_roles": claims.get("scope", {}).get("roles", []),
            "issued_scope_keys": claims.get("scope", {}).get("access_scope_keys", []),
        },
    )

    invalid_status, _, _ = http_call(
        f"{base_url}/api/v1/threads",
        headers={"Authorization": "Bearer invalid.token.value"},
    )
    expired_status, _, _ = http_call(
        f"{base_url}/api/v1/threads",
        headers={"Authorization": f"Bearer {token_for_scope(EXPECTED_PRODUCTION_SCOPE, values, expired=True)}"},
    )
    check(
        "invalid_and_expired_jwt_rejected",
        invalid_status == 401 and expired_status == 401,
        {"invalid_status": invalid_status, "expired_status": expired_status},
    )

    engineer_scope = {**EXPECTED_PRODUCTION_SCOPE, "user_id": "security_engineer", "roles": ["engineer"]}
    engineer_token = token_for_scope(engineer_scope, values)
    admin_status, _, _ = http_call(
        f"{base_url}/api/v1/knowledge-documents",
        headers={"Authorization": f"Bearer {engineer_token}"},
    )
    check("knowledge_admin_rbac", admin_status == 403, admin_status)

    valid_headers = {"Authorization": f"Bearer {token}"}
    threads_status, threads_body, _ = http_call(f"{base_url}/api/v1/threads", headers=valid_headers)
    threads = json.loads(threads_body) if threads_status == 200 else []
    ownership_status = None
    if threads:
        ownership_status, _, _ = http_call(
            f"{base_url}/api/v1/threads/{parse.quote(threads[0]['thread_id'], safe='')}",
            headers={"Authorization": f"Bearer {engineer_token}"},
        )
    check(
        "cross_user_thread_isolation",
        threads_status == 200 and bool(threads) and ownership_status == 404,
        {"thread_list_status": threads_status, "cross_user_status": ownership_status},
    )

    seed_anonymous_status, _, _ = http_call(f"{base_url}/api/v1/demo/seed", method="POST")
    seed_authorized_status, _, _ = http_call(
        f"{base_url}/api/v1/demo/seed",
        method="POST",
        headers=valid_headers,
    )
    check(
        "production_demo_seed_closed",
        seed_anonymous_status == 401 and seed_authorized_status == 404,
        {"anonymous_status": seed_anonymous_status, "authorized_status": seed_authorized_status},
    )

    image_id, access = find_accessible_image_id(containers, base_url, token)
    asset_evidence: dict[str, Any] = {"image_found": bool(image_id)}
    asset_passed = bool(image_id and access)
    if image_id and access:
        expires_at = datetime.fromisoformat(access["expires_at"])
        remaining = int((expires_at - datetime.now(UTC)).total_seconds())
        signed_url = parse.urljoin(f"{base_url}/", access["url"])
        parsed_url = parse.urlsplit(signed_url)
        signed_status, _, _ = http_call(signed_url)
        unauthorized_scope = {**engineer_scope, "access_scope_keys": ["unauthorized_scope"]}
        denied_status, _, _ = http_call(
            f"{base_url}/api/v1/assets/{parse.quote(image_id, safe='')}/access",
            headers={"Authorization": f"Bearer {token_for_scope(unauthorized_scope, values)}"},
        )
        asset_passed = 240 <= remaining <= 330 and signed_status == 200 and denied_status == 403
        asset_evidence.update(
            {
                "remaining_seconds": remaining,
                "signed_path": parsed_url.path,
                "signed_status": signed_status,
                "unauthorized_status": denied_status,
            }
        )
        if args.wait_asset_expiry:
            sleep_for = max(0, remaining + 3)
            time.sleep(sleep_for)
            expired_asset_status, _, _ = http_call(signed_url)
            asset_evidence["expired_status"] = expired_asset_status
            asset_passed = asset_passed and expired_asset_status == 403
    check("authorized_asset_link_and_expiry", asset_passed, asset_evidence)

    surfaces: dict[str, bytes] = {"git_tracked_files": tracked_file_bytes(root)}
    for service in APP_SERVICES:
        inspected = containers.get(service)
        if not inspected:
            continue
        name = inspected["Name"].lstrip("/")
        surfaces[f"{service}_logs"] = run("docker", "logs", "--since", "24h", name)
        image = inspected["Image"]
        surfaces[f"{service}_image_config"] = run("docker", "image", "inspect", image)
    if "web" in containers:
        web_name = containers["web"]["Name"].lstrip("/")
        surfaces["frontend_static_bundle"] = run(
            "docker",
            "exec",
            web_name,
            "tar",
            "-C",
            "/usr/share/nginx/html",
            "-cf",
            "-",
            ".",
        )
    hits = secret_hits(values, surfaces)
    check("secrets_absent_from_public_surfaces", not hits, hits)

    if args.test_rate_limit:
        statuses = []
        for _ in range(12):
            status_code, _, _ = http_call(
                f"{base_url}/api/v1/auth/demo-token",
                method="POST",
                headers={"X-Demo-Access-Key": "deliberately-invalid-access-key"},
            )
            statuses.append(status_code)
        check("demo_access_rate_limit", 429 in statuses, statuses)

    failures = [item["name"] for item in checks if not item["passed"]]
    report = {
        "schema": "semikb-t948-security-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "checks": checks,
        "passed": not failures,
        "failed_checks": failures,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe security report to {args.output}")
    else:
        print(serialized, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.output.decode("utf-8", "replace"))
        raise
