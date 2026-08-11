from __future__ import annotations

import pytest

from semikb.contracts.models import ActorScope


def test_hybrid_retrieval_keeps_only_current_authorized_evidence(seeded_services) -> None:
    _, _, retrieval, _, _ = seeded_services
    evidence, trace = retrieval.search("ETCH-03 Chamber B 清腔后首片异常，当前 SOP 怎么要求？", ActorScope())

    assert "SOP-ETCH-03-R2-001" in trace.final_evidence_ids
    assert "SOP-ETCH-03-R1-001" not in trace.final_evidence_ids
    assert trace.actor_user_id == "demo_engineer"
    assert all(chunk.lifecycle.value == "published" for chunk in evidence)


def test_text_can_recall_authorized_real_image_reference(seeded_services) -> None:
    _, _, retrieval, _, _ = seeded_services
    _, trace = retrieval.search("有没有 ETCH-03 Chamber B 的边缘环状缺陷晶圆图？", ActorScope())

    assert "IMG-FA-ETCH-03-2026-004" in trace.image_asset_ids
    access = retrieval.asset_access("IMG-FA-ETCH-03-2026-004", ActorScope())
    assert access["url"].endswith("/preview")


def test_asset_access_cannot_bypass_product_scope(seeded_services) -> None:
    _, _, retrieval, _, _ = seeded_services
    outsider = ActorScope(user_id="outsider", products=["P-BETA"])

    with pytest.raises(PermissionError):
        retrieval.asset_access("IMG-FA-ETCH-03-2026-004", outsider)
