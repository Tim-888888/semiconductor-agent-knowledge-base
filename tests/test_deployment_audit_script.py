from __future__ import annotations

import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_docker_host.sh"


def test_docker_host_audit_has_no_infrastructure_mutation_commands() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    forbidden = re.compile(
        r"\bdocker\s+(?:compose\s+)?"
        r"(?:up|down|start|stop|restart|kill|rm|rmi|pull|push|build|run|create)\b"
    )

    assert forbidden.search(script) is None
    assert "docker network create" not in script
    assert "docker network rm" not in script
    assert "docker volume create" not in script
    assert "docker volume rm" not in script


def test_docker_host_audit_does_not_disclose_environment_or_secrets() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert ".Config.Env" not in script
    assert "printenv" not in script
    assert "/environ" not in script
    assert "docker secret" not in script


def test_docker_host_audit_supports_centos7_and_detects_external_changes() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "MOUNTPOINTS" not in script
    assert "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT" in script
    assert "containers_unchanged=true" in script
    assert "containers_unchanged=false" in script
    assert script.count("docker_snapshot >") == 2
