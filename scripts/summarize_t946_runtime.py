"""Summarize credential-safe host and Docker samples emitted by T9-4.6."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

UNIT_FACTORS = {
    "B": 1,
    "kB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
}


def parse_size(raw: str) -> float:
    match = re.fullmatch(r"\s*([0-9.]+)\s*([A-Za-z]+)\s*", raw)
    if not match or match.group(2) not in UNIT_FACTORS:
        raise ValueError(f"unsupported Docker size: {raw}")
    return float(match.group(1)) * UNIT_FACTORS[match.group(2)]


def parse_percent(raw: str) -> float:
    return float(raw.strip().removesuffix("%"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def summarize(sample_dir: Path) -> dict[str, Any]:
    host = load_jsonl(sample_dir / "host.jsonl")
    containers = load_jsonl(sample_dir / "containers.jsonl")
    if not host or not containers:
        raise ValueError("runtime samples are incomplete")
    by_container: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in containers:
        by_container[str(item["Name"])].append(item)
    container_summary = {}
    for name, samples in sorted(by_container.items()):
        memory_usage = [parse_size(str(item["MemUsage"]).split("/")[0]) for item in samples]
        container_summary[name] = {
            "sample_count": len(samples),
            "max_cpu_percent": max(parse_percent(str(item["CPUPerc"])) for item in samples),
            "max_memory_bytes": int(max(memory_usage)),
            "max_memory_percent": max(
                parse_percent(str(item["MemPerc"])) for item in samples
            ),
            "max_pids": max(int(item["PIDs"]) for item in samples),
        }
    host_available = [int(item["mem_available_kib"]) * 1024 for item in host]
    root_available = [int(item["root_available_bytes"]) for item in host]
    return {
        "schema": "semikb-t946-runtime-summary-v1",
        "host": {
            "sample_count": len(host),
            "min_available_memory_bytes": min(host_available),
            "min_root_available_bytes": min(root_available),
            "root_available_delta_bytes": root_available[-1] - root_available[0],
            "max_load_1m": max(float(item["load_1m"]) for item in host),
            "swap_configured": any(int(item["swap_total_kib"]) > 0 for item in host),
        },
        "containers": container_summary,
    }


def main() -> None:
    args = parse_args()
    report = summarize(args.sample_dir.resolve())
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")
        print(f"wrote credential-safe runtime summary to {output}")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
