"""Focused tests for the false-positive actor-injection replay readiness check.

Covers the readiness contract for issue #3300: a valid condition is ``ready``,
an omitted/empty injection is ``not_available``, and malformed inputs or missing
provenance fail closed as ``blocked`` with actionable blocker messages.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.false_positive_injection_readiness import (
    FALSE_POSITIVE_INJECTION_READINESS_SCHEMA_VERSION,
    FALSE_POSITIVE_PERTURBATION_FAMILY,
    REQUIRED_PROVENANCE_FIELDS,
    STATUS_BLOCKED,
    STATUS_NOT_AVAILABLE,
    STATUS_READY,
    check_false_positive_injection_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
READY_FIXTURE = (
    REPO_ROOT
    / "tests/fixtures/benchmark/false_positive_actor_injection/replay_condition_ready.yaml"
)
CLI_PATH = REPO_ROOT / "scripts/benchmark/check_false_positive_injection_readiness.py"
CLI_BLOCKED_EXIT_CODE = 3


def _valid_provenance() -> dict[str, object]:
    """Return a complete provenance mapping for the false-positive family."""
    return {
        "scenario_id": "occluded_emergence_episode_0000",
        "seed": 2755,
        "planner_mode": "trace_derived",
        "perturbation_family": FALSE_POSITIVE_PERTURBATION_FAMILY,
        "execution_mode": "readiness_check_only",
    }


def _ready_spec() -> dict[str, object]:
    """Return a fully valid false-positive injection replay-condition spec."""
    return {
        **_valid_provenance(),
        "false_positive_positions": [[2.0, 3.0]],
        "false_positive_velocities": [[0.0, 0.0]],
        "false_positive_ids": ["fp_close"],
    }


class TestReadyCondition:
    """Specs that should be constructible into a replay condition."""

    def test_valid_spec_is_ready(self) -> None:
        """A complete spec with injected actors is ready."""
        readiness = check_false_positive_injection_readiness(_ready_spec())
        assert readiness.status == STATUS_READY
        assert readiness.is_ready is True
        assert readiness.injected_actor_count == 1
        assert readiness.noise_profile == FALSE_POSITIVE_PERTURBATION_FAMILY
        assert readiness.blockers == []

    def test_ready_payload_round_trips_to_json(self) -> None:
        """The ready verdict serializes with schema and issue provenance."""
        payload = check_false_positive_injection_readiness(_ready_spec()).to_dict()
        assert payload["schema_version"] == FALSE_POSITIVE_INJECTION_READINESS_SCHEMA_VERSION
        assert payload["issue"] == 3300
        assert payload["status"] == STATUS_READY
        # provenance is echoed back for the report contract
        assert payload["provenance"]["scenario_id"] == "occluded_emergence_episode_0000"
        json.dumps(payload)  # must be serializable

    def test_multiple_injected_actors_counted(self) -> None:
        """Injected-actor count reflects all requested false positives."""
        spec = {**_valid_provenance(), "false_positive_positions": [[2.0, 3.0], [4.0, 5.0]]}
        readiness = check_false_positive_injection_readiness(spec)
        assert readiness.status == STATUS_READY
        assert readiness.injected_actor_count == 2


class TestNotAvailable:
    """Specs with no injected actors: accepted-unavailable, not blocked."""

    def test_omitted_injection_is_not_available(self) -> None:
        """No false-positive actors requested mirrors the #2927 unavailable state."""
        readiness = check_false_positive_injection_readiness(_valid_provenance())
        assert readiness.status == STATUS_NOT_AVAILABLE
        assert readiness.injected_actor_count == 0
        assert readiness.is_ready is False
        assert readiness.blockers == []

    def test_empty_positions_is_not_available(self) -> None:
        """An empty positions list is accepted-unavailable, not blocked."""
        spec = {**_valid_provenance(), "false_positive_positions": []}
        readiness = check_false_positive_injection_readiness(spec)
        assert readiness.status == STATUS_NOT_AVAILABLE


class TestFailClosed:
    """Malformed inputs or missing provenance must fail closed."""

    @pytest.mark.parametrize("missing", REQUIRED_PROVENANCE_FIELDS)
    def test_missing_each_provenance_field_blocks(self, missing: str) -> None:
        """Dropping any required provenance field fails closed."""
        spec = _ready_spec()
        del spec[missing]
        readiness = check_false_positive_injection_readiness(spec)
        assert readiness.status == STATUS_BLOCKED
        assert any(missing in blocker for blocker in readiness.blockers)

    def test_empty_string_provenance_blocks(self) -> None:
        """Whitespace-only provenance is treated as missing."""
        spec = {**_ready_spec(), "scenario_id": "   "}
        readiness = check_false_positive_injection_readiness(spec)
        assert readiness.status == STATUS_BLOCKED
        assert any("scenario_id" in blocker for blocker in readiness.blockers)

    def test_wrong_perturbation_family_blocks(self) -> None:
        """A mismatched perturbation_family fails closed."""
        spec = {**_ready_spec(), "perturbation_family": "missed_detection"}
        readiness = check_false_positive_injection_readiness(spec)
        assert readiness.status == STATUS_BLOCKED
        assert any("perturbation_family" in blocker for blocker in readiness.blockers)

    def test_malformed_positions_block(self) -> None:
        """A bad injection-input shape must fail closed, not raise out."""
        spec = {**_valid_provenance(), "false_positive_positions": [1.0, 2.0, 3.0]}
        readiness = check_false_positive_injection_readiness(spec)
        assert readiness.status == STATUS_BLOCKED
        assert any("injection input" in blocker for blocker in readiness.blockers)

    def test_velocity_count_mismatch_blocks(self) -> None:
        """Velocity/position count mismatch fails closed via the canonical spec."""
        spec = {
            **_valid_provenance(),
            "false_positive_positions": [[1.0, 2.0]],
            "false_positive_velocities": [[0.0, 0.0], [1.0, 0.0]],
        }
        readiness = check_false_positive_injection_readiness(spec)
        assert readiness.status == STATUS_BLOCKED

    def test_non_mapping_spec_raises(self) -> None:
        """A non-mapping spec is a programming error, not a readiness state."""
        with pytest.raises(TypeError, match="must be a mapping"):
            check_false_positive_injection_readiness(["not", "a", "mapping"])  # type: ignore[arg-type]


class TestCli:
    """CLI wrapper exit-code and payload contract."""

    def test_cli_ready_fixture_exits_zero(self) -> None:
        """The CLI exits 0 and reports ready for the tracked fixture."""
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), str(READY_FIXTURE)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["status"] == STATUS_READY
        assert payload["injected_actor_count"] == 1

    def test_cli_blocked_spec_exits_nonzero(self, tmp_path: Path) -> None:
        """The CLI fails closed with a non-zero exit on a blocked spec."""
        bad = tmp_path / "blocked.yaml"
        bad.write_text(
            "false_positive_injection:\n  false_positive_positions:\n    - [1.0, 2.0]\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), str(bad)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.returncode == CLI_BLOCKED_EXIT_CODE
        payload = json.loads(result.stdout)
        assert payload["status"] == STATUS_BLOCKED
        assert payload["blockers"]

    def test_cli_malformed_yaml_fails_closed(self, tmp_path: Path) -> None:
        """Unparseable YAML fails closed with a blocked payload, not a traceback."""
        bad = tmp_path / "malformed.yaml"
        bad.write_text("false_positive_injection: [unterminated\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), str(bad)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.returncode == CLI_BLOCKED_EXIT_CODE, result.stderr
        payload = json.loads(result.stdout)
        assert payload["status"] == STATUS_BLOCKED
        assert any("failed to load spec file" in blocker for blocker in payload["blockers"])

    def test_cli_non_mapping_spec_fails_closed(self, tmp_path: Path) -> None:
        """A spec that parses to a non-mapping fails closed instead of raising."""
        bad = tmp_path / "scalar.yaml"
        bad.write_text("42\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), str(bad)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.returncode == CLI_BLOCKED_EXIT_CODE, result.stderr
        payload = json.loads(result.stdout)
        assert payload["status"] == STATUS_BLOCKED
        assert any("must contain a mapping" in blocker for blocker in payload["blockers"])
