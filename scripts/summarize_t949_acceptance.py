"""Combine T9-4.9 component reports into one credential-safe final verdict."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.verify_t947_restore import compare_snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(evidence_dir: Path, baseline_path: Path) -> dict[str, Any]:
    demo = load(evidence_dir / "final-demo.json")
    security = load(evidence_dir / "security.json")
    offline = load(evidence_dir / "offline.json")
    baseline = load(baseline_path)
    current = load(evidence_dir / "storage-state.json")
    storage_comparison = compare_snapshots(baseline, current)
    worker = (evidence_dir / "worker-ping.txt").read_text(encoding="utf-8")
    health = load(evidence_dir / "health.json")
    checks = [
        {"name": "final_demo", "passed": bool(demo.get("passed"))},
        {"name": "security", "passed": bool(security.get("passed"))},
        {"name": "offline_bundle", "passed": bool(offline.get("passed"))},
        {
            "name": "storage_invariants",
            "passed": storage_comparison["matched"]
            and bool(current.get("retrieval_smoke", {}).get("passed")),
        },
        {"name": "worker", "passed": "pong" in worker.lower()},
        {"name": "health", "passed": health.get("status") == "ok"},
    ]
    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema": "semikb-t949-final-acceptance-v1",
        "captured_at": datetime.now(UTC).isoformat(),
        "checks": checks,
        "storage_comparison": storage_comparison,
        "passed": not failures,
        "failed_checks": failures,
    }


def main() -> None:
    args = parse_args()
    report = summarize(args.evidence_dir.resolve(), args.baseline.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote T9-4.9 final verdict to {output}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
