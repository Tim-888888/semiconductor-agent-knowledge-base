from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from semikb.rag_ingestion.service import IngestionService
from semikb.storage.memory import DemoStore

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "fixtures" / "demo_corpus.json"
GOLDEN_V2_PATH = ROOT / "data" / "golden_sets" / "demo_v2.json"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_demo_documents_have_explicit_governance_and_unique_versions() -> None:
    corpus = load_json(CORPUS_PATH)
    documents = corpus["documents"]
    required = {
        "document_id",
        "revision",
        "document_type",
        "approval_status",
        "lifecycle",
        "effective_at",
        "source_kind",
        "source_uri",
        "source_license",
        "access_scope_key",
        "fab",
        "product",
    }

    identities = {(item["document_id"], item["revision"]) for item in documents}
    assert len(documents) == len(identities)
    assert all(required.issubset(item) for item in documents)
    assert all(item["source_kind"] == "synthetic" for item in documents)
    assert all(item["source_uri"].startswith("synthetic://semikb/") for item in documents)
    assert all(item["source_license"] == "CC0-1.0" for item in documents)
    assert all(item["access_scope_key"] for item in documents)


def test_recipe_versions_have_an_explicit_supersession_chain() -> None:
    documents = load_json(CORPUS_PATH)["documents"]
    recipes = sorted(
        (item for item in documents if item["document_type"] == "recipe"),
        key=lambda item: item["revision"],
    )

    assert [item["revision"] for item in recipes] == ["V2.2", "V2.3"]
    assert recipes[0]["lifecycle"] == "expired"
    assert recipes[1]["lifecycle"] == "published"
    assert recipes[1]["supersedes_revision"] == "V2.2"


def test_wafer_image_files_exist_open_and_match_declared_hashes() -> None:
    documents = load_json(CORPUS_PATH)["documents"]
    images = [image for item in documents for image in item.get("images", [])]

    assert len(images) >= 2
    for image in images:
        path = (ROOT / image["source_path"]).resolve()
        assert path.is_relative_to((ROOT / "data" / "assets").resolve())
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert sha256(path) == image["sha256"]


def test_spc_csv_has_required_scope_limits_and_anomaly_labels() -> None:
    documents = load_json(CORPUS_PATH)["documents"]
    spc_document = next(item for item in documents if item["document_type"] == "spc_summary")
    asset = spc_document["data_assets"][0]
    path = ROOT / asset["path"]

    assert sha256(path) == asset["sha256"]
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "timestamp",
        "lot_id",
        "wafer_id",
        "tool_id",
        "chamber",
        "recipe_id",
        "recipe_version",
        "value",
        "center_line",
        "ucl",
        "lcl",
        "is_ooc",
        "anomaly_type",
    }
    assert len(rows) == 120
    assert required.issubset(rows[0])
    assert all(row["tool_id"] == "ETCH-03" and row["chamber"] == "B" for row in rows)
    assert sum(row["is_ooc"] == "true" for row in rows) == 5


def test_golden_v2_has_explicit_scope_outcomes_and_valid_chunk_references() -> None:
    payload = load_json(GOLDEN_V2_PATH)
    cases = payload["cases"]
    store = DemoStore()
    IngestionService(store).seed_demo_corpus(CORPUS_PATH)

    assert payload["dataset_version"] == "demo-v2"
    assert len(cases) >= 8
    assert all(case["actor_scope"] for case in cases)
    assert all(case["expected_outcome"] in {"evidence", "no_evidence"} for case in cases)
    assert all(case["failure_labels"] for case in cases)
    assert any(case["expected_outcome"] == "no_evidence" for case in cases)
    for case in cases:
        for chunk_id in case["expected_chunk_ids"]:
            assert chunk_id in store.chunks

    covered_tags = {tag for case in cases for tag in case["tags"]}
    assert {"recipe", "spc", "image_text", "expired", "acl", "no_answer"}.issubset(
        covered_tags
    )
