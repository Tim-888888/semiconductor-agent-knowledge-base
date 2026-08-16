from __future__ import annotations

import pytest

from semikb.contracts.evaluation_governance import (
    CreateEvaluationReleaseFreezeRequest,
    EvaluationReleaseFreezeStatus,
    RegisterEvaluationDatasetRequest,
)
from semikb.contracts.models import (
    EvaluationCase,
    EvaluationDatasetPurpose,
    EvaluationLeakageStatus,
)
from semikb.demo_factory import demo_actor_scope


def _dataset_request(
    version: str,
    purpose: EvaluationDatasetPurpose,
) -> RegisterEvaluationDatasetRequest:
    return RegisterEvaluationDatasetRequest(
        dataset_version=version,
        source_kind="process-separated-test",
        description=f"{purpose.value} evaluation split",
        purpose=purpose,
        source_snapshot_hash="a" * 64,
        leakage_status=EvaluationLeakageStatus.CLEARED,
        seal=purpose is EvaluationDatasetPurpose.HOLDOUT,
        cases=[
            EvaluationCase(
                case_id=f"{purpose.value}-case-1",
                question="What is the first action after an ETCH-03 pressure alarm?",
                expected_chunk_ids=["SOP-ETCH-03-R2-002"],
                actor_scope=demo_actor_scope(),
            )
        ],
    )


def test_holdout_requires_release_freeze_and_opens_exact_snapshot(seeded_services) -> None:
    _, _, _, _, evaluation = seeded_services
    versions = {
        purpose: f"t9446d-{purpose.value}-v1"
        for purpose in EvaluationDatasetPurpose
    }
    for purpose, version in versions.items():
        evaluation.register_dataset(_dataset_request(version, purpose))

    with pytest.raises(ValueError, match="frozen into a release snapshot"):
        evaluation.create_run(versions[EvaluationDatasetPurpose.HOLDOUT])

    freeze = evaluation.create_release_freeze(
        CreateEvaluationReleaseFreezeRequest(
            release_version="t9446d-release-v1",
            source_commit="abcdef1234567",
            publication_batch_ids=["corpus-publication-completed"],
            development_dataset_version=versions[EvaluationDatasetPurpose.DEVELOPMENT],
            calibration_dataset_version=versions[EvaluationDatasetPurpose.CALIBRATION],
            regression_dataset_version=versions[EvaluationDatasetPurpose.REGRESSION],
            holdout_dataset_version=versions[EvaluationDatasetPurpose.HOLDOUT],
            notes="Code and retrieval configuration frozen before holdout opening.",
        ),
        created_by="admin",
    )
    run = evaluation.create_run(versions[EvaluationDatasetPurpose.HOLDOUT])
    opened = evaluation.list_release_freezes()[0]

    assert freeze.status is EvaluationReleaseFreezeStatus.FROZEN
    assert opened.status is EvaluationReleaseFreezeStatus.OPENED
    assert run.release_freeze_id == freeze.freeze_id
    assert run.release_freeze_hash == freeze.freeze_hash
    assert run.dataset_opened_at is not None

    with pytest.raises(ValueError, match="frozen into a release snapshot"):
        evaluation.create_run(versions[EvaluationDatasetPurpose.HOLDOUT])


def test_registered_dataset_versions_are_immutable_and_idempotent(seeded_services) -> None:
    _, _, _, _, evaluation = seeded_services
    request = _dataset_request("t9446d-development-idempotent-v1", EvaluationDatasetPurpose.DEVELOPMENT)
    first = evaluation.register_dataset(request)
    second = evaluation.register_dataset(request)
    assert second.dataset_hash == first.dataset_hash

    changed = request.model_copy(update={"description": "changed after registration"})
    with pytest.raises(ValueError, match="different content"):
        evaluation.register_dataset(changed)
