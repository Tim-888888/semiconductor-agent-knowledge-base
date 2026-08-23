from __future__ import annotations

from pathlib import Path

from scripts.verify_t948_security import (
    decode_payload,
    encode_hs256,
    is_root_user,
    read_env,
    secret_hits,
)


def test_t948_env_reader_and_secret_scan_never_return_secret_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "API_PASSWORD=credential-value-123\nPUBLIC_SETTING=visible\n",
        encoding="utf-8",
    )

    values = read_env(env_path)
    hits = secret_hits(values, {"frontend": b"prefix credential-value-123 suffix"})

    assert hits == [{"surface": "frontend", "secret_key": "API_PASSWORD"}]
    assert "credential-value-123" not in repr(hits)


def test_t948_stdlib_jwt_helper_encodes_expected_scope() -> None:
    payload = {"sub": "security-user", "scope": {"roles": ["engineer"]}, "exp": 2_000_000_000}

    token = encode_hs256(payload, "test-secret-that-is-long-enough")

    assert decode_payload(token) == payload


def test_t948_root_user_detection_covers_docker_defaults() -> None:
    assert is_root_user("")
    assert is_root_user("0")
    assert is_root_user("root:root")
    assert not is_root_user("10001:10001")
    assert not is_root_user("nginx")
