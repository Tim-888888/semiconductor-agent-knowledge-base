from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.deployment.generate_server_env import generate, read_env
from semikb.deployment.seed_demo_corpus import load_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_server_env_generator_excludes_workstation_and_ssh_credentials(tmp_path: Path) -> None:
    source = tmp_path / ".env"
    source.write_text(
        "\n".join(
            (
                "MONGODB_URI=mongodb://old:secret@192.168.10.100:27017",
                "DEPLOY_SSH_PASSWORD=must-not-copy",
                "CLOSEAI_API_KEY=close-key",
                "QWEN_API_KEY=qwen-key",
                "RERANK_API_KEY=rerank-key",
                "MINERU_API_KEY=mineru-key",
                "ALIYUN_WEB_MCP_API_KEY=web-key",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / ".env.server"

    generate(source, output)
    values = read_env(output)

    assert values["APP_ENV"] == "production"
    assert values["DEMO_MODE"] == "false"
    assert values["MAX_UPLOAD_MIB"] == "100"
    assert values["MONGODB_URI"].startswith("mongodb://semikb_app:")
    assert "@mongodb:27017/semikb?authSource=semikb" in values["MONGODB_URI"]
    assert values["MILVUS_URI"] == "http://milvus:19530"
    assert values["MINIO_ENDPOINT"] == "minio:9000"
    assert values["REDIS_URL"].endswith("@redis:6379/0")
    assert len(values["JWT_SECRET"]) >= 64
    assert len(values["DEMO_ACCESS_KEY"]) >= 32
    rendered = output.read_text(encoding="utf-8")
    assert "192.168.10.100" not in rendered
    assert "DEPLOY_SSH" not in rendered
    assert "must-not-copy" not in rendered
    assert values["CLOSEAI_API_KEY"] == "close-key"


def test_server_env_generator_refuses_accidental_overwrite(tmp_path: Path) -> None:
    source = tmp_path / ".env"
    source.write_text("QWEN_API_KEY=test\n", encoding="utf-8")
    output = tmp_path / ".env.server"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate(source, output)


def test_production_compose_is_single_node_hardened_and_exposes_only_web() -> None:
    compose = (ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")

    for image in (
        "mongo:8.2.6",
        "milvusdb/milvus:v2.5.5",
        "quay.io/coreos/etcd:v3.5.18",
        "minio/minio:RELEASE.2023-03-20T20-16-18Z",
        "quay.io/minio/minio:RELEASE.2024-12-18T13-15-44Z",
        "redis:7.4.10-alpine",
    ):
        assert image in compose
    assert compose.count("ports:") == 1
    assert '"80:8080"' in compose
    assert "internal: true" in compose
    assert "assets:" in compose
    assert "--requirepass" in compose
    assert "authorizationEnabled: false/authorizationEnabled: true" in compose
    assert "grep -q 'authorizationEnabled: true' /milvus/configs/milvus.yaml" in compose
    assert "./deploy/milvus/milvus.yaml:/milvus/configs/milvus.yaml" not in compose
    assert not (ROOT / "deploy/milvus/milvus.yaml").exists()
    assert "--concurrency=1" in compose
    assert compose.count("mem_limit:") == 10
    assert "MINIO_PUBLIC_BASE_URL: /objects" in compose
    web_section = compose.split("  web:\n", maxsplit=1)[1].split("\nnetworks:", maxsplit=1)[0]
    assert "read_only: true" in web_section
    assert "cap_drop:\n      - ALL" in web_section
    assert "/tmp:size=64m,mode=1777" in web_section


def test_production_python_image_uses_hashed_dependency_lock() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert "COPY requirements.lock ./" in dockerfile
    assert "pip install --require-hashes -r requirements.lock" in dockerfile
    assert "pip install ." not in dockerfile
    assert "PYTHONPATH=/app/src" in dockerfile
    assert "PIP_DEFAULT_TIMEOUT=180" in dockerfile
    assert "PIP_RETRIES=10" in dockerfile
    assert "--hash=sha256:" in lock
    assert "langgraph==" in lock
    assert "pymilvus==" in lock


def test_deployment_scripts_keep_restore_and_stage_guards() -> None:
    deploy = (ROOT / "scripts/deployment/deploy.sh").read_text(encoding="utf-8")
    backup = (ROOT / "scripts/deployment/backup_cold.sh").read_text(encoding="utf-8")
    restore = (ROOT / "scripts/deployment/restore_cold.sh").read_text(encoding="utf-8")
    cleanup = (ROOT / "scripts/deployment/cleanup_t947_restore.sh").read_text(encoding="utf-8")
    preflight = (ROOT / "scripts/deployment/host_preflight.sh").read_text(encoding="utf-8")

    assert "host_preflight.sh" in deploy
    assert "pull --policy missing" in deploy
    assert 'if [[ "$offline" == "true" ]]' in deploy
    assert "config --images" in deploy
    assert "Offline deployment is missing image" in deploy
    assert "semikb.storage.provisioning" in deploy
    assert 'docker wait "$milvus_init_id"' in deploy
    assert "--seed-demo" in deploy
    assert "seed_demo_corpus --apply" in deploy
    assert "up -d api worker" in deploy
    assert "up -d --force-recreate web" in deploy
    assert "-f docker-compose.yml -f docker-compose.prod.yml" not in deploy
    assert "--confirm-empty-target" in restore
    assert "--env" in restore
    assert "--target-data-root" in restore
    assert "--restore-env" in restore
    assert "--project-name" in restore
    assert "restore-t947-independent" in restore
    assert "Refusing restore because target is not empty" in restore
    assert 'install -m 0600 "$env_backup" .env' not in restore
    assert "docker compose --env-file" not in restore
    assert "docker compose down" not in restore
    assert "T947_BACKUP_CONFIRM" in backup
    assert 'stop web api worker' in backup
    assert 'stop mongodb redis milvus etcd minio milvus-minio' in backup
    assert "T947_CLEANUP_CONFIRM" in cleanup
    assert 'label=com.docker.compose.project=$project_name' in cleanup
    assert 'target_data_root" != "$live_data_root' in cleanup
    assert "Security-group rules must still be verified" in preflight
    assert "docker system prune" not in "\n".join((deploy, backup, restore, cleanup, preflight))
    assert "source .env" not in "\n".join((deploy, backup, restore, cleanup, preflight))
    assert 'source "$env_backup"' not in restore


def test_production_compose_can_bind_an_independent_restore_environment() -> None:
    compose = (ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")

    assert compose.count("env_file: ${SEMIKB_APP_ENV_FILE:-.env}") == 2
    assert 'ports:\n      - "80:8080"' in compose


def test_web_image_runs_nginx_as_non_root_on_an_unprivileged_port() -> None:
    dockerfile = (ROOT / "web/Dockerfile").read_text(encoding="utf-8")
    nginx = (ROOT / "web/nginx.conf").read_text(encoding="utf-8")
    nginx_main = (ROOT / "web/nginx-main.conf").read_text(encoding="utf-8")

    assert "USER 101:101" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert 'ENTRYPOINT ["nginx"]' in dockerfile
    assert "listen 8080;" in nginx
    assert "pid /tmp/nginx.pid;" in nginx_main
    assert "client_body_temp_path /tmp/client_temp;" in nginx_main


def test_t946_runtime_scripts_are_bounded_and_restart_requires_double_confirmation() -> None:
    sampler = (ROOT / "scripts/deployment/sample_t946_runtime.sh").read_text(encoding="utf-8")
    restart = (ROOT / "scripts/deployment/restart_t946_services.sh").read_text(encoding="utf-8")

    assert "SAMPLES <= 900" in sampler
    assert "INTERVAL_SECONDS <= 60" in sampler
    assert "docker stats --no-stream" in sampler
    assert ".State.Health" not in sampler
    assert "T946_RESTART_CONFIRM" in restart
    assert '"--apply"' in restart
    assert "docker compose" in restart
    assert " restart \"$service\"" in restart
    assert 'state.get("Health", {}).get("Status", "none")' in restart
    assert '[[ "$ping_output" == *pong* ]]' in restart
    assert "| grep -q pong" not in restart
    assert "{{if .State.Health}}" not in restart
    assert "{{if .Config.Healthcheck}}" not in restart
    assert "docker compose down" not in restart
    assert "docker system prune" not in restart
    assert "docker volume rm" not in restart


def test_alibaba_linux_installer_is_platform_gated_and_non_destructive() -> None:
    installer = (ROOT / "scripts/deployment/install_moby_alinux4.sh").read_text(encoding="utf-8")

    assert '"${ID:-}" != "alinux"' in installer
    assert "dnf install -y moby docker-compose-plugin" in installer
    assert "systemctl enable --now docker" in installer
    assert "remove" not in installer
    assert "rm -rf" not in installer


def test_linux_deployment_scripts_are_executable_in_git() -> None:
    scripts = sorted((ROOT / "scripts/deployment").glob("*.sh"))

    for script in scripts:
        relative = script.relative_to(ROOT).as_posix()
        stage = subprocess.check_output(
            ["git", "ls-files", "--stage", "--", relative],
            cwd=ROOT,
            text=True,
        )
        assert stage.startswith("100755 "), f"{relative} must be executable after clone"


def test_deployment_seed_uses_the_governed_synthetic_corpus() -> None:
    documents = load_corpus(ROOT / "data/fixtures/demo_corpus.json")

    assert len(documents) >= 5
    assert all(item["source_kind"] == "synthetic" for item in documents)
    assert all(item["source_license"] == "CC0-1.0" for item in documents)


def test_frontend_does_not_mix_jwt_with_minio_query_authentication() -> None:
    api_source = (ROOT / "web/src/api.ts").read_text(encoding="utf-8")

    assert 'token && !access.url.startsWith("/objects/")' in api_source


def test_nginx_disables_buffering_for_agent_sse() -> None:
    nginx = (ROOT / "web/nginx.conf").read_text(encoding="utf-8")

    assert "messages/stream$" in nginx
    assert "proxy_buffering off;" in nginx
    assert "proxy_cache off;" in nginx
    assert "gzip off;" in nginx
    assert 'add_header X-Accel-Buffering "no" always;' in nginx
