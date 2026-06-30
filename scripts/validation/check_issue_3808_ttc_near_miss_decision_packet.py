#!/usr/bin/env python3
"""Fail-closed decision-packet checker for issue #3808 TTC near-miss diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PACKET_PATH = Path("configs/benchmarks/issue_3808_ttc_near_miss_decision_packet.yaml")
SCHEMA_VERSION = "issue-3808-ttc-near-miss-decision-packet.v1"
EXPECTED_FIXTURE_STATUS = {
    "closing": "ok",
    "opening": "no-approaching-pairs",
    "missing-timing": "unsupported-inputs",
    "unsupported-trajectory": "unsupported-inputs",
}
EXPECTED_EVIDENCE_STATUS = "ready_for_decision"


class PacketError(ValueError):
    """Raised for fail-closed validation failures."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    _require(isinstance(value, Mapping), f"{key} must be a mapping")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PacketError("packet must be YAML mapping")
    return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_relative(path_value: Any, key: str) -> Path:
    _require(isinstance(path_value, str), f"{key} must be a string path")
    value = path_value.strip()
    _require(value != "", f"{key} must be non-empty")
    path = Path(value)
    _require(not path.is_absolute(), f"{key} must be repo-relative")
    return path


def _run_renderer(packet: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    renderer = _require_mapping(packet, "renderer")
    script_rel = _repo_relative(renderer.get("script"), "renderer.script")
    script = repo_root / script_rel
    _require(script.is_file(), "renderer script missing")

    raw_args = renderer.get("args")
    _require(isinstance(raw_args, list), "renderer.args must be a list")

    completed = subprocess.run(
        [sys.executable, str(script), *(str(arg) for arg in raw_args)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    _require(completed.returncode == 0, "renderer CLI command must exit 0")
    stdout = completed.stdout.strip()
    _require(stdout != "", "renderer CLI returned empty output")

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise PacketError(f"renderer output must be JSON: {exc}") from exc


def validate_packet(packet: Mapping[str, Any], repo_root: Path | None = None) -> dict[str, Any]:
    """Validate issue #3808 decision packet contract and fixture outputs."""
    repo_root = repo_root or _repo_root()

    _require(packet.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(packet.get("issue") == 3808, "issue must be 3808")

    claim_boundary = str(packet.get("claim_boundary", "")).lower()
    _require(
        "full benchmark campaign run" in claim_boundary and "paper/dissertation" in claim_boundary,
        "claim_boundary must explicitly reject benchmark campaign and paper/dissertation claim",
    )

    execution = _require_mapping(packet, "execution_boundary")
    _require(
        execution.get("run_benchmark") is False, "execution_boundary.run_benchmark must be false"
    )
    _require(
        execution.get("compute_submit_authorized") is False,
        "execution_boundary.compute_submit_authorized must be false",
    )
    _require(
        str(execution.get("slurm_job_id", "")).lower() == "not_submitted",
        "execution_boundary.slurm_job_id must be not_submitted",
    )

    expected = _require_mapping(packet, "expected_fixtures")
    _require(
        set(expected) == set(EXPECTED_FIXTURE_STATUS),
        "expected_fixtures must exactly match issue #3808 fixture set",
    )
    for name in EXPECTED_FIXTURE_STATUS:
        entry = _require_mapping(expected, name)
        _require(
            entry.get("diagnostic_status") == EXPECTED_FIXTURE_STATUS[name],
            f"expected_fixtures[{name}].diagnostic_status must be {EXPECTED_FIXTURE_STATUS[name]}",
        )
        _require(
            entry.get("evidence_status") == EXPECTED_EVIDENCE_STATUS,
            f"expected_fixtures[{name}].evidence_status must be {EXPECTED_EVIDENCE_STATUS}",
        )

    rendered = _run_renderer(packet, repo_root)
    _require(rendered.get("issue") == 3808, "renderer issue mismatch")

    fixtures = rendered.get("fixtures")
    _require(isinstance(fixtures, Mapping), "renderer payload fixtures must be a mapping")

    fixture_names = set(fixtures)
    _require(
        fixture_names == set(EXPECTED_FIXTURE_STATUS),
        "renderer fixture set mismatch",
    )

    observed = {name: payload.get("diagnostic_status") for name, payload in fixtures.items()}
    _require(observed == EXPECTED_FIXTURE_STATUS, "renderer fixture diagnostic_status mismatch")

    claim_boundary_text = str(rendered.get("claim_boundary", "")).lower()
    _require(
        "paper-facing claim" in claim_boundary_text,
        "rendered payload must preserve claim boundary",
    )

    return {
        "status": "ok",
        "issue": 3808,
        "schema_version": SCHEMA_VERSION,
        "fixture_count": len(fixtures),
        "fixtures": {name: {"diagnostic_status": status} for name, status in observed.items()},
        "render_command_exit_code": 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_PACKET_PATH,
        help="Path to issue #3808 decision packet.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the issue #3808 packet checker CLI."""
    args = _build_parser().parse_args(argv)
    packet_path = args.packet

    try:
        packet = _load_yaml(packet_path)
        report = validate_packet(packet)
    except PacketError as exc:
        if args.json:
            print(json.dumps({"status": "malformed", "error": str(exc)}, sort_keys=True))
        else:
            print(f"error: {exc}")
        return 2

    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print("issue_3808 decision packet checks ok")
        print(f"fixtures: {report['fixture_count']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
