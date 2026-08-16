"""Contracts for immutable evaluation datasets and release freezes."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from semikb.contracts.models import (
    EvaluationCase,
    EvaluationDatasetPurpose,
    EvaluationLeakageStatus,
    new_id,
    utc_now,
)


class StrictEvaluationGovernanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationReleaseFreezeStatus(StrEnum):
    FROZEN = "frozen"
    OPENED = "opened"


class RegisterEvaluationDatasetRequest(StrictEvaluationGovernanceModel):
    dataset_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    source_kind: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=4000)
    purpose: EvaluationDatasetPurpose
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    leakage_status: EvaluationLeakageStatus
    cases: list[EvaluationCase] = Field(min_length=1, max_length=1000)
    seal: bool = False

    @model_validator(mode="after")
    def validate_registration(self) -> RegisterEvaluationDatasetRequest:
        if self.purpose is EvaluationDatasetPurpose.HOLDOUT:
            if not self.seal:
                raise ValueError("Holdout datasets must be sealed when registered.")
            if self.leakage_status is not EvaluationLeakageStatus.CLEARED:
                raise ValueError("Holdout datasets require a cleared leakage review.")
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique within a dataset.")
        return self


class CreateEvaluationReleaseFreezeRequest(StrictEvaluationGovernanceModel):
    release_version: str = Field(min_length=1, max_length=160)
    source_commit: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    publication_batch_ids: list[str] = Field(min_length=1, max_length=100)
    development_dataset_version: str
    calibration_dataset_version: str
    regression_dataset_version: str
    holdout_dataset_version: str
    notes: str = Field(default="", max_length=4000)


class EvaluationReleaseFreeze(StrictEvaluationGovernanceModel):
    freeze_schema_version: Literal["semikb-evaluation-release-freeze-v1"] = (
        "semikb-evaluation-release-freeze-v1"
    )
    freeze_id: str = Field(default_factory=lambda: new_id("eval_freeze"))
    release_version: str
    source_commit: str
    publication_batch_ids: list[str]
    dataset_hashes: dict[str, str]
    holdout_dataset_version: str
    holdout_dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_config: dict[str, object]
    component_versions: dict[str, str]
    freeze_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: EvaluationReleaseFreezeStatus = EvaluationReleaseFreezeStatus.FROZEN
    created_by: str
    notes: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    opened_at: datetime | None = None
