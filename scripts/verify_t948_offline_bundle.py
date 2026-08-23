"""Verify and optionally load a T9-4.8 offline migration bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--reload-aliases", action="store_true")
    parser.add_argument("--verify-docker", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checksum_entries(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        digest, name = line.split(None, 1)
        result[name.strip()] = digest
    return result


def docker_image_id(tag: str) -> str | None:
    try:
        raw = subprocess.check_output(
            ["docker", "image", "inspect", tag],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    return json.loads(raw)[0]["Id"]


def main() -> None:
    args = parse_args()
    bundle = args.bundle_dir.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    expected = checksum_entries(bundle / "SHA256SUMS")
    checksum_results: dict[str, dict[str, Any]] = {}
    for name, digest in expected.items():
        actual = sha256_file(bundle / name)
        checksum_results[name] = {
            "expected": digest,
            "actual": actual,
            "matched": digest == actual,
        }
    checksums_passed = all(item["matched"] for item in checksum_results.values())
    if not checksums_passed:
        raise SystemExit("Offline bundle checksum verification failed.")

    if args.reload_aliases:
        for record in manifest["images"]:
            alias = record["offline_alias"]
            if not alias.startswith("semikb-offline/"):
                raise SystemExit(f"Refusing to remove non-offline alias: {alias}")
            subprocess.run(
                ["docker", "image", "rm", alias],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    if args.load or args.reload_aliases:
        subprocess.check_call(["docker", "load", "-i", str(bundle / "images.tar")])

    docker_results: list[dict[str, Any]] = []
    if args.verify_docker or args.load or args.reload_aliases:
        for record in manifest["images"]:
            runtime_id = docker_image_id(record["runtime_tag"])
            alias_id = docker_image_id(record["offline_alias"])
            docker_results.append(
                {
                    "service": record["service"],
                    "runtime_tag": record["runtime_tag"],
                    "offline_alias": record["offline_alias"],
                    "expected_image_id": record["image_id"],
                    "runtime_image_id": runtime_id,
                    "alias_image_id": alias_id,
                    "matched": runtime_id == record["image_id"] and alias_id == record["image_id"],
                }
            )
    docker_passed = not docker_results or all(item["matched"] for item in docker_results)
    report = {
        "schema": "semikb-t948-offline-verification-v1",
        "verified_at": datetime.now(UTC).isoformat(),
        "source_commit": manifest["source_commit"],
        "checksums_passed": checksums_passed,
        "checksum_results": checksum_results,
        "docker_verified": bool(docker_results),
        "docker_passed": docker_passed,
        "docker_results": docker_results,
        "passed": checksums_passed and docker_passed,
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"wrote offline verification report to {args.output}")
    else:
        print(serialized, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
