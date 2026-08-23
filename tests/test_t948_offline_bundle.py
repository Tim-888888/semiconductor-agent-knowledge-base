from __future__ import annotations

import json
from pathlib import Path

from scripts.build_t948_offline_bundle import SERVICES, runtime_image_tag, sha256_file
from scripts.verify_t948_offline_bundle import checksum_entries


def test_t948_checksum_parser_and_digest_match(tmp_path: Path) -> None:
    payload = tmp_path / "images.tar"
    payload.write_bytes(b"offline-image-payload")
    digest = sha256_file(payload)
    sums = tmp_path / "SHA256SUMS"
    sums.write_text(f"{digest}  images.tar\n", encoding="ascii")

    assert checksum_entries(sums) == {"images.tar": digest}


def test_t948_offline_bundle_covers_every_production_service() -> None:
    assert SERVICES == (
        "etcd",
        "milvus-minio",
        "milvus",
        "mongodb",
        "minio",
        "redis",
        "milvus-init",
        "api",
        "worker",
        "web",
    )


def test_t948_runtime_image_tag_supports_pulled_and_compose_built_images() -> None:
    config = {
        "name": "semikb",
        "services": {
            "mongodb": {"image": "mongo:8.2.6"},
            "api": {"build": {"context": "."}},
        },
    }

    assert runtime_image_tag(config, "mongodb") == "mongo:8.2.6"
    assert runtime_image_tag(config, "api") == "semikb-api:latest"


def test_t948_bundle_uses_an_explicit_git_archive_instead_of_the_worktree() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/build_t948_offline_bundle.py").read_text(
        encoding="utf-8"
    )

    assert 'run("git", "rev-parse", args.source_ref' in source
    assert '"git", "archive", "--format=tar.gz"' in source
    assert '"git", "diff"' not in source


def test_t948_bundle_manifest_contract_contains_no_credentials(tmp_path: Path) -> None:
    manifest = {
        "schema": "semikb-t948-offline-bundle-v1",
        "source_commit": "a" * 40,
        "images": [
            {
                "service": "api",
                "runtime_tag": "semikb-api:latest",
                "offline_alias": f"semikb-offline/api:{'a' * 12}",
                "image_id": f"sha256:{'b' * 64}",
            }
        ],
    }

    rendered = json.dumps(manifest)

    assert ".env" not in rendered
    assert "password" not in rendered.lower()
    assert "api_key" not in rendered.lower()
