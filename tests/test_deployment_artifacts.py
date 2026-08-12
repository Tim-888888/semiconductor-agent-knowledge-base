from __future__ import annotations

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
    assert '"80:80"' in compose
    assert "internal: true" in compose
    assert "assets:" in compose
    assert "--requirepass" in compose
    assert "authorizationEnabled: true" in (ROOT / "deploy/milvus/milvus.yaml").read_text(encoding="utf-8")
    assert "--concurrency=1" in compose
    assert compose.count("mem_limit:") == 10
    assert "MINIO_PUBLIC_BASE_URL: /objects" in compose


def test_deployment_scripts_keep_restore_and_stage_guards() -> None:
    deploy = (ROOT / "scripts/deployment/deploy.sh").read_text(encoding="utf-8")
    restore = (ROOT / "scripts/deployment/restore_cold.sh").read_text(encoding="utf-8")
    preflight = (ROOT / "scripts/deployment/host_preflight.sh").read_text(encoding="utf-8")

    assert "host_preflight.sh" in deploy
    assert "pull --policy missing" in deploy
    assert "semikb.storage.provisioning" in deploy
    assert 'docker wait "$milvus_init_id"' in deploy
    assert "--seed-demo" in deploy
    assert "seed_demo_corpus --apply" in deploy
    assert "--confirm-empty-target" in restore
    assert "--env" in restore
    assert "Refusing restore because target is not empty" in restore
    assert "Security-group rules must still be verified" in preflight
    assert "docker system prune" not in "\n".join((deploy, restore, preflight))
    assert "source .env" not in "\n".join((deploy, restore, preflight))
    assert 'source "$env_backup"' not in restore


def test_alibaba_linux_installer_is_platform_gated_and_non_destructive() -> None:
    installer = (ROOT / "scripts/deployment/install_moby_alinux4.sh").read_text(encoding="utf-8")

    assert '"${ID:-}" != "alinux"' in installer
    assert "dnf install -y moby docker-compose-plugin" in installer
    assert "systemctl enable --now docker" in installer
    assert "remove" not in installer
    assert "rm -rf" not in installer


def test_deployment_seed_uses_the_governed_synthetic_corpus() -> None:
    documents = load_corpus(ROOT / "data/fixtures/demo_corpus.json")

    assert len(documents) >= 5
    assert all(item["source_kind"] == "synthetic" for item in documents)
    assert all(item["source_license"] == "CC0-1.0" for item in documents)


def test_frontend_does_not_mix_jwt_with_minio_query_authentication() -> None:
    api_source = (ROOT / "web/src/api.ts").read_text(encoding="utf-8")

    assert 'token && !access.url.startsWith("/objects/")' in api_source
