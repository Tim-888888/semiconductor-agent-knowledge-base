"""Stable, safe errors exposed by the ingestion boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IngestErrorCode(StrEnum):
    UNSUPPORTED_FORMAT = "INGEST_UNSUPPORTED_FORMAT"
    FILE_TYPE_MISMATCH = "INGEST_FILE_TYPE_MISMATCH"
    INVALID_OFFICE_CONTAINER = "INGEST_INVALID_OFFICE_CONTAINER"
    ZIP_BOMB_SUSPECTED = "INGEST_ZIP_BOMB_SUSPECTED"
    CORRUPT_DOCUMENT = "INGEST_CORRUPT_DOCUMENT"
    ENCRYPTED_DOCUMENT = "INGEST_ENCRYPTED_DOCUMENT"
    TEXT_DECODE_FAILED = "INGEST_TEXT_DECODE_FAILED"
    DOCUMENT_LIMIT_EXCEEDED = "INGEST_DOCUMENT_LIMIT_EXCEEDED"
    PARSER_NOT_CONFIGURED = "INGEST_PARSER_NOT_CONFIGURED"
    PARSER_UNAVAILABLE = "INGEST_PARSER_UNAVAILABLE"
    PARSER_TIMEOUT = "INGEST_PARSER_TIMEOUT"
    PARSE_FAILED = "INGEST_PARSE_FAILED"
    EMPTY_PARSE_RESULT = "INGEST_EMPTY_PARSE_RESULT"
    ASSET_EXTRACTION_FAILED = "INGEST_ASSET_EXTRACTION_FAILED"
    QUALITY_GATE_FAILED = "INGEST_QUALITY_GATE_FAILED"
    CONTRACT_VIOLATION = "INGEST_CONTRACT_VIOLATION"


class ErrorStage(StrEnum):
    VALIDATING = "validating"
    PARSING = "parsing"
    QUALITY_CHECK = "quality_check"


class RetryPolicy(StrEnum):
    NEVER = "never"
    AFTER_SOURCE_REPLACEMENT = "after_source_replacement"
    AFTER_POLICY_CHANGE = "after_policy_change"
    AFTER_CONFIGURATION = "after_configuration"
    AUTOMATIC_BOUNDED = "automatic_bounded"
    AFTER_REVIEW = "after_review"
    AFTER_DEPLOYMENT_FIX = "after_deployment_fix"


@dataclass(frozen=True, slots=True)
class ErrorDescriptor:
    http_status: int
    stage: ErrorStage
    retry_policy: RetryPolicy
    quarantine: bool


ERROR_CATALOG: dict[IngestErrorCode, ErrorDescriptor] = {
    IngestErrorCode.UNSUPPORTED_FORMAT: ErrorDescriptor(
        415, ErrorStage.VALIDATING, RetryPolicy.NEVER, False
    ),
    IngestErrorCode.FILE_TYPE_MISMATCH: ErrorDescriptor(
        422, ErrorStage.VALIDATING, RetryPolicy.AFTER_SOURCE_REPLACEMENT, False
    ),
    IngestErrorCode.INVALID_OFFICE_CONTAINER: ErrorDescriptor(
        422, ErrorStage.VALIDATING, RetryPolicy.AFTER_SOURCE_REPLACEMENT, False
    ),
    IngestErrorCode.ZIP_BOMB_SUSPECTED: ErrorDescriptor(
        422, ErrorStage.VALIDATING, RetryPolicy.NEVER, True
    ),
    IngestErrorCode.CORRUPT_DOCUMENT: ErrorDescriptor(
        422, ErrorStage.PARSING, RetryPolicy.AFTER_SOURCE_REPLACEMENT, True
    ),
    IngestErrorCode.ENCRYPTED_DOCUMENT: ErrorDescriptor(
        422, ErrorStage.PARSING, RetryPolicy.AFTER_SOURCE_REPLACEMENT, True
    ),
    IngestErrorCode.TEXT_DECODE_FAILED: ErrorDescriptor(
        422, ErrorStage.PARSING, RetryPolicy.AFTER_SOURCE_REPLACEMENT, True
    ),
    IngestErrorCode.DOCUMENT_LIMIT_EXCEEDED: ErrorDescriptor(
        413, ErrorStage.VALIDATING, RetryPolicy.AFTER_POLICY_CHANGE, False
    ),
    IngestErrorCode.PARSER_NOT_CONFIGURED: ErrorDescriptor(
        503, ErrorStage.PARSING, RetryPolicy.AFTER_CONFIGURATION, False
    ),
    IngestErrorCode.PARSER_UNAVAILABLE: ErrorDescriptor(
        503, ErrorStage.PARSING, RetryPolicy.AUTOMATIC_BOUNDED, False
    ),
    IngestErrorCode.PARSER_TIMEOUT: ErrorDescriptor(
        504, ErrorStage.PARSING, RetryPolicy.AUTOMATIC_BOUNDED, False
    ),
    IngestErrorCode.PARSE_FAILED: ErrorDescriptor(
        422, ErrorStage.PARSING, RetryPolicy.AFTER_REVIEW, True
    ),
    IngestErrorCode.EMPTY_PARSE_RESULT: ErrorDescriptor(
        422, ErrorStage.QUALITY_CHECK, RetryPolicy.AFTER_REVIEW, True
    ),
    IngestErrorCode.ASSET_EXTRACTION_FAILED: ErrorDescriptor(
        422, ErrorStage.QUALITY_CHECK, RetryPolicy.AFTER_REVIEW, True
    ),
    IngestErrorCode.QUALITY_GATE_FAILED: ErrorDescriptor(
        422, ErrorStage.QUALITY_CHECK, RetryPolicy.AFTER_REVIEW, True
    ),
    IngestErrorCode.CONTRACT_VIOLATION: ErrorDescriptor(
        500, ErrorStage.QUALITY_CHECK, RetryPolicy.AFTER_DEPLOYMENT_FIX, True
    ),
}


class IngestError(RuntimeError):
    """Error with a stable code and a user-safe message."""

    def __init__(self, code: IngestErrorCode, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.descriptor = ERROR_CATALOG[code]
