from __future__ import annotations

import json
from pathlib import Path


def test_clarification_evaluation_set_is_frozen_and_not_a_production_input() -> None:
    path = Path("data/evaluation_specs/t9451_clarification_scenarios_v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset_version"] == "t9-4.5.1-clarification-v1"
    assert payload["purpose"] == "holdout_acceptance"
    assert payload["production_usage"] == "forbidden"
    assert len(payload["scenarios"]) >= 8
    assert all(item["turns"] for item in payload["scenarios"])
