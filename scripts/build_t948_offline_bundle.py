"""Export a fixed-image, credential-free T9-4.8 migration bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SERVICES = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument("--compose-file", type=Path, default=Path("docker-compose.prod.yml"))
    parser.add_argument("--source-ref", default="HEAD")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def run(*command: str, cwd: Path, text: bool = True) -> str | bytes:
    return subprocess.check_output(command, cwd=cwd, text=text, stderr=subprocess.STDOUT)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source_ref(root: Path, source_ref: str) -> str:
    return str(
        run(
            "git",
            "rev-parse",
            "--verify",
            f"{source_ref}^{{commit}}",
            cwd=root,
        )
    ).strip()


def compose_config(root: Path, env_path: Path, compose_path: Path) -> dict[str, Any]:
    raw = run(
        "docker",
        "compose",
        "--env-file",
        str(env_path),
        "-f",
        str(compose_path),
        "config",
        "--format",
        "json",
        cwd=root,
    )
    return json.loads(raw)


def runtime_image_tag(config: dict[str, Any], service: str) -> str:
    service_config = config["services"][service]
    if image := service_config.get("image"):
        return str(image)
    if "build" in service_config:
        project_name = str(config.get("name") or "").strip()
        if not project_name:
            raise SystemExit(
                f"Compose service {service!r} uses build but the project has no fixed name."
            )
        return f"{project_name}-{service}:latest"
    raise SystemExit(f"Compose service {service!r} defines neither image nor build.")


def image_record(service: str, runtime_tag: str, source_ref: str, root: Path) -> dict[str, Any]:
    inspected = json.loads(run("docker", "image", "inspect", runtime_tag, cwd=root))[0]
    alias = f"semikb-offline/{service}:{source_ref[:12]}"
    subprocess.check_call(
        ["docker", "image", "tag", inspected["Id"], alias],
        cwd=root,
    )
    return {
        "service": service,
        "runtime_tag": runtime_tag,
        "offline_alias": alias,
        "image_id": inspected["Id"],
        "repo_digests": sorted(inspected.get("RepoDigests") or []),
        "os": inspected.get("Os"),
        "architecture": inspected.get("Architecture"),
        "size_bytes": inspected.get("Size"),
    }


def main() -> None:
    args = parse_args()
    if not args.apply:
        raise SystemExit("Refusing to export images without --apply.")
    root = Path(__file__).resolve().parents[1]
    env_path = args.env.resolve()
    compose_path = args.compose_file.resolve()
    output = args.output_dir.resolve()
    source_ref = resolve_source_ref(root, args.source_ref)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = compose_config(root, env_path, compose_path)
    missing_services = [service for service in SERVICES if service not in config["services"]]
    if missing_services:
        raise SystemExit(f"Compose is missing required services: {', '.join(missing_services)}")

    records = [
        image_record(service, runtime_image_tag(config, service), source_ref, root)
        for service in SERVICES
    ]
    unique_image_sizes = {
        record["image_id"]: int(record["size_bytes"] or 0) for record in records
    }
    required_free = sum(unique_image_sizes.values()) + 2 * 1024**3
    available = shutil.disk_usage(output.parent).free
    if available < required_free:
        raise SystemExit(
            f"Insufficient free space: need at least {required_free} bytes, have {available}."
        )

    image_tags = sorted(
        {tag for record in records for tag in (record["runtime_tag"], record["offline_alias"])}
    )
    images_archive = output / "images.tar"
    subprocess.check_call(["docker", "save", "-o", str(images_archive), *image_tags], cwd=root)

    source_archive = output / "source.tar.gz"
    subprocess.check_call(
        ["git", "archive", "--format=tar.gz", "-o", str(source_archive), source_ref],
        cwd=root,
    )
    compose_sha = sha256_file(compose_path)
    manifest = {
        "schema": "semikb-t948-offline-bundle-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_commit": source_ref,
        "compose_sha256": compose_sha,
        "images": records,
        "artifacts": {},
    }
    manifest_path = output / "manifest.json"
    readme = output / "README.txt"
    readme.write_text(
        "\n".join(
            (
                "SemiAtlas T9-4.8 offline migration bundle",
                "",
                "1. Copy this entire directory to the target host.",
                "2. Extract source.tar.gz into /opt/semiconductor-agent-knowledge-base/.",
                "3. Create the target .env separately; this bundle contains no credentials or business data.",
                "4. Run: python3 scripts/verify_t948_offline_bundle.py --bundle-dir <bundle> --load --verify-docker",
                "5. Run: ./scripts/deployment/deploy.sh --offline",
                "",
                "Restore business data from a separately governed cold backup after verifying the target paths.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for path in (images_archive, source_archive, readme):
        manifest["artifacts"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checksum_paths = (images_archive, source_archive, manifest_path, readme)
    checksums = output / "SHA256SUMS"
    checksums.write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "source_commit": source_ref,
                "image_count": len(records),
                "unique_image_count": len(unique_image_sizes),
                "images_archive_bytes": images_archive.stat().st_size,
                "images_archive_sha256": sha256_file(images_archive),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
