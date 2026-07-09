#!/usr/bin/env python3
"""Tests for derive_issue_4206_trace_verified_failure_mechanisms."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "scripts" / "analysis"),
)

from derive_issue_4206_trace_verified_failure_mechanisms import (
    _derive_mechanism_label,
    _is_collision,
    _is_failure,
    _is_timeout,
    build_mechanism_sidecar,
)

from robot_sf.benchmark.failure_mechanism_taxonomy import (
    MECHANISM_SCHEMA_VERSION,
    validate_failure_mechanism_record,
)


class _TmpDir:
    """Minimal temp directory context manager for tests."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self._created: list[Path] = []

    def __enter__(self) -> Path:
        import tempfile

        self.path = Path(tempfile.mkdtemp(prefix="rsf_test_mech_"))
        return self.path

    def __exit__(self, *args: object) -> None:
        if self.path is not None:
            for p in self._created:
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                self.path.rmdir()
            except OSError:
                pass


def _make_episode(
    *,
    outcome: dict[str, Any] | None = None,
    event_ledger: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    episode_id: str = "test-ep-001",
    scenario_id: str = "test_scenario",
    seed: int = 42,
    algo: str = "test_planner",
) -> dict[str, Any]:
    """Build a synthetic episode record for mechanism derivation tests."""
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": algo,
        "outcome": outcome
        or {"collision_event": False, "route_complete": True, "timeout_event": False},
        "event_ledger": event_ledger
        or {
            "collision_events": [],
            "exact_events": {
                "collision": False,
                "goal_reached": True,
                "invalid_run": False,
                "timeout": False,
            },
            "surrogate_events": {
                "clearance_breach": False,
                "late_evasive": False,
                "near_miss": False,
                "occlusion_near_miss": False,
                "oscillation": False,
            },
        },
        "metrics": metrics
        or {
            "collisions": 0,
            "near_misses": 0,
            "stalled_time": 0.0,
            "avg_speed": 1.0,
            "path_efficiency": 0.9,
            "time_to_goal_norm": 0.1,
            "total_collision_count": 0,
        },
    }


class TestOutcomeHelpers:
    """Tests for outcome classification helpers."""

    def test_is_failure_success(self) -> None:
        assert not _is_failure({"route_complete": True})

    def test_is_failure_collision(self) -> None:
        assert _is_failure({"route_complete": False, "collision_event": True})

    def test_is_failure_timeout(self) -> None:
        assert _is_failure({"route_complete": False, "timeout_event": True})

    def test_is_collision_true(self) -> None:
        assert _is_collision({"collision_event": True})

    def test_is_collision_false(self) -> None:
        assert not _is_collision({"collision_event": False})

    def test_is_timeout_true(self) -> None:
        assert _is_timeout({"timeout_event": True})

    def test_is_timeout_false(self) -> None:
        assert not _is_timeout({"timeout_event": False})


class TestDeriveMechanismLabel:
    """Tests for mechanism label derivation rules."""

    def test_success_episode_returns_none(self) -> None:
        ep = _make_episode(outcome={"route_complete": True, "collision_event": False})
        result = _derive_mechanism_label(ep)
        assert result is None

    def test_collision_with_late_evasive(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": True,
                "timeout_event": False,
            },
            event_ledger={
                "collision_events": [{"collision_time": 10.0}],
                "exact_events": {"collision": True},
                "surrogate_events": {
                    "clearance_breach": True,
                    "late_evasive": True,
                },
            },
            metrics={
                "total_collision_count": 1,
                "near_misses": 2,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "proxemic_or_clearance_tradeoff"
        assert result["mechanism_confidence"] == "supported_hypothesis"
        assert result["mechanism_evidence_mode"] == "paired_trace"

    def test_collision_basic(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": True,
                "timeout_event": False,
            },
            metrics={"total_collision_count": 1},
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "proxemic_or_clearance_tradeoff"

    def test_timeout_static_deadlock_low_speed(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            metrics={
                "avg_speed": 0.001,
                "stalled_time": 5.0,
                "path_efficiency": 0.3,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "static_deadlock_or_local_minimum"

    def test_timeout_stalled(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            metrics={
                "avg_speed": 0.5,
                "stalled_time": 15.0,
                "path_efficiency": 0.4,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "static_deadlock_or_local_minimum"

    def test_timeout_low_path_efficiency(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            metrics={
                "avg_speed": 0.8,
                "stalled_time": 2.0,
                "path_efficiency": 0.1,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "static_deadlock_or_local_minimum"

    def test_timeout_with_oscillation(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            event_ledger={
                "surrogate_events": {"oscillation": True},
            },
            metrics={
                "avg_speed": 0.8,
                "stalled_time": 2.0,
                "path_efficiency": 0.5,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "dynamic_phase_or_order_sensitivity"

    def test_timeout_moderate(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            metrics={
                "avg_speed": 0.8,
                "stalled_time": 2.0,
                "path_efficiency": 0.5,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "time_budget_artifact"

    def test_failure_insufficient_surfaces(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": False,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "unknown"
        assert result["mechanism_confidence"] == "unknown"

    def test_schema_version_always_correct(self) -> None:
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": True,
                "timeout_event": False,
            }
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        assert result["mechanism_schema_version"] == MECHANISM_SCHEMA_VERSION


class TestBuildMechanismSidecar:
    """Tests for the full mechanism-sidecar build pipeline."""

    def _make_campaign(self, tmp: Path) -> Path:
        """Create a minimal campaign directory with episode JSONL."""
        campaign = tmp / "test_campaign"
        runs = campaign / "runs"

        # Successful planner
        p1 = runs / "goal__differential_drive"
        p1.mkdir(parents=True, exist_ok=True)
        episodes = [
            _make_episode(
                episode_id=f"goal-scenario1-seed{s}",
                outcome={"route_complete": True, "collision_event": False, "timeout_event": False},
            )
            for s in range(5)
        ]
        with (p1 / "episodes.jsonl").open("w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        # Failed planner with collisions and timeouts
        p2 = runs / "social_force__differential_drive"
        p2.mkdir(parents=True, exist_ok=True)
        episodes = [
            _make_episode(
                episode_id=f"sf-scenario1-seed{s}",
                algo="social_force",
                outcome={
                    "route_complete": False,
                    "collision_event": True,
                    "timeout_event": False,
                },
                metrics={"total_collision_count": 1, "near_misses": 2},
            )
            for s in range(3)
        ]
        episodes.append(
            _make_episode(
                episode_id="sf-scenario1-seed3-timeout",
                algo="social_force",
                outcome={
                    "route_complete": False,
                    "collision_event": False,
                    "timeout_event": True,
                },
                metrics={
                    "avg_speed": 0.001,
                    "stalled_time": 5.0,
                    "path_efficiency": 0.3,
                },
            )
        )
        with (p2 / "episodes.jsonl").open("w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        (campaign / "campaign_manifest.json").write_text(
            json.dumps(
                {
                    "campaign_id": "test_campaign",
                    "planners": [
                        {"key": "goal", "status": "ok"},
                        {"key": "social_force", "status": "ok"},
                    ],
                }
            )
        )
        return campaign

    def test_build_sidecar_produces_files(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        summary = build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        assert summary["status"] == "completed"
        assert (output_dir / "mechanism_labels.csv").is_file()
        assert (output_dir / "mechanism_labels.jsonl").is_file()
        assert (output_dir / "label_coverage.json").is_file()
        assert (output_dir / "selection_manifest.json").is_file()
        assert (output_dir / "README.md").is_file()

    def test_build_sidecar_failure_count(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        summary = build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        assert summary["failure_count"] == 4
        assert summary["labeled_count"] >= 4

    def test_mechanism_labels_validate(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        jsonl_path = output_dir / "mechanism_labels.jsonl"
        with jsonl_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                mechanism_fields = {k: record[k] for k in record if k.startswith("mechanism_")}
                validate_failure_mechanism_record(mechanism_fields)

    def test_missing_campaign_root_blocks(self, tmp_path: Path) -> None:
        campaign = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"

        summary = build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        # Missing campaign must block with an explicit status, not raise and not
        # silently emit a vacuous "completed" packet.
        assert summary["status"] == "blocked_missing_input_artifacts"
        assert "reason" in summary
        assert not (output_dir / "mechanism_labels.csv").exists()
        assert not (output_dir / "README.md").exists()
        assert (output_dir / "input_status.json").is_file()

    def test_empty_campaign_blocks(self, tmp_path: Path) -> None:
        campaign = tmp_path / "empty_campaign"
        runs = campaign / "runs"
        planner_run = runs / "goal__differential_drive"
        planner_run.mkdir(parents=True)
        (planner_run / "episodes.jsonl").write_text("", encoding="utf-8")
        output_dir = tmp_path / "output"

        summary = build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        # A campaign dir present but with zero episode rows must also block.
        assert summary["status"] == "blocked_missing_input_artifacts"
        assert not (output_dir / "mechanism_labels.csv").exists()


class TestGuardedPPOExclusion:
    """Tests for guarded_ppo exclusion in sidecar build."""

    def test_guarded_ppo_skipped(self, tmp_path: Path) -> None:
        campaign = tmp_path / "test_campaign"
        runs = campaign / "runs"
        gp = runs / "guarded_ppo__differential_drive"
        gp.mkdir(parents=True, exist_ok=True)

        ep = _make_episode(
            algo="guarded_ppo",
            outcome={
                "route_complete": False,
                "collision_event": True,
                "timeout_event": False,
            },
            metrics={"total_collision_count": 1},
        )
        ep["status"] = "not_available"
        with (gp / "episodes.jsonl").open("w") as f:
            f.write(json.dumps(ep) + "\n")

        (campaign / "campaign_manifest.json").write_text(
            json.dumps({"planners": [{"key": "guarded_ppo", "status": "not_available"}]})
        )

        output_dir = tmp_path / "output"
        summary = build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")
        assert summary["failure_count"] == 0
        assert summary["labeled_count"] == 0


class TestFailClosedMetrics:
    """Tests for fail-closed behavior when metrics are missing."""

    def test_timeout_missing_avg_speed_fails_closed(self) -> None:
        """Timeout with missing avg_speed must NOT label as static_deadlock."""
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            metrics={
                "avg_speed": None,
                "stalled_time": 5.0,
                "path_efficiency": 0.3,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        # Missing avg_speed should not trigger static_deadlock_or_local_minimum
        assert result["mechanism_label"] != "static_deadlock_or_local_minimum"

    def test_timeout_missing_path_efficiency_fails_closed(self) -> None:
        """Timeout with missing path_efficiency must NOT label as static_deadlock."""
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            metrics={
                "avg_speed": 0.8,
                "stalled_time": 2.0,
                "path_efficiency": None,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        # Missing path_efficiency should not trigger static_deadlock_or_local_minimum
        assert result["mechanism_label"] != "static_deadlock_or_local_minimum"

    def test_timeout_both_metrics_missing_fails_closed(self) -> None:
        """Timeout with both metrics missing must label as unknown."""
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            metrics={
                "avg_speed": None,
                "stalled_time": 2.0,
                "path_efficiency": None,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "unknown"
        assert result["mechanism_confidence"] == "unknown"
        assert "insufficient metrics" in result["mechanism_caveat"]

    def test_timeout_present_metrics_works_normally(self) -> None:
        """Timeout with present metrics should work as before."""
        ep = _make_episode(
            outcome={
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
            metrics={
                "avg_speed": 0.001,
                "stalled_time": 5.0,
                "path_efficiency": 0.3,
            },
        )
        result = _derive_mechanism_label(ep)
        assert result is not None
        validate_failure_mechanism_record(result)
        assert result["mechanism_label"] == "static_deadlock_or_local_minimum"


class TestNewOutputFiles:
    """Tests for new output files: input_audit, crosscut_interpretation, SHA256SUMS, label_coverage.md."""

    def _make_campaign(self, tmp: Path) -> Path:
        """Create a minimal campaign directory with episode JSONL."""
        campaign = tmp / "test_campaign"
        runs = campaign / "runs"

        # Successful planner
        p1 = runs / "goal__differential_drive"
        p1.mkdir(parents=True, exist_ok=True)
        episodes = [
            _make_episode(
                episode_id=f"goal-scenario1-seed{s}",
                outcome={"route_complete": True, "collision_event": False, "timeout_event": False},
            )
            for s in range(5)
        ]
        with (p1 / "episodes.jsonl").open("w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        # Failed planner with collisions
        p2 = runs / "social_force__differential_drive"
        p2.mkdir(parents=True, exist_ok=True)
        episodes = [
            _make_episode(
                episode_id=f"sf-scenario1-seed{s}",
                algo="social_force",
                outcome={
                    "route_complete": False,
                    "collision_event": True,
                    "timeout_event": False,
                },
                metrics={"total_collision_count": 1, "near_misses": 2},
            )
            for s in range(3)
        ]
        with (p2 / "episodes.jsonl").open("w") as f:
            for ep in episodes:
                f.write(json.dumps(ep) + "\n")

        (campaign / "campaign_manifest.json").write_text(
            json.dumps(
                {
                    "campaign_id": "test_campaign",
                    "planners": [
                        {"key": "goal", "status": "ok"},
                        {"key": "social_force", "status": "ok"},
                    ],
                }
            )
        )
        return campaign

    def test_input_audit_files_created(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        assert (output_dir / "input_audit.json").is_file()
        assert (output_dir / "input_audit.md").is_file()

    def test_input_audit_content(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        with (output_dir / "input_audit.json").open() as f:
            audit = json.loads(f.readline())
            assert audit["status"] == "ok"
            assert audit["total_episodes"] == 8
            assert audit["failure_episodes"] == 3

    def test_crosscut_interpretation_files_created(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        assert (output_dir / "crosscut_interpretation.json").is_file()
        assert (output_dir / "crosscut_interpretation.md").is_file()

    def test_crosscut_interpretation_content(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        with (output_dir / "crosscut_interpretation.json").open() as f:
            interp = json.loads(f.readline())
            assert "trace_verified_mechanisms" in interp
            assert "unresolved_mechanisms" in interp
            assert "guarded_ppo" in interp["excluded_arms"]

    def test_sha256sums_created(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        assert (output_dir / "SHA256SUMS").is_file()

    def test_sha256sums_content(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        with (output_dir / "SHA256SUMS").open() as f:
            lines = f.readlines()
            assert len(lines) > 0
            for line in lines:
                parts = line.strip().split("  ")
                assert len(parts) == 2
                assert len(parts[0]) == 64  # SHA256 hex length

    def test_label_coverage_md_created(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        assert (output_dir / "label_coverage.md").is_file()

    def test_label_coverage_md_content(self, tmp_path: Path) -> None:
        campaign = self._make_campaign(tmp_path)
        output_dir = tmp_path / "output"

        build_mechanism_sidecar(campaign, output_dir, "2026-01-01T00:00:00Z")

        content = (output_dir / "label_coverage.md").read_text()
        assert "Label Coverage Report" in content
        assert "Total episodes" in content
        assert "Label Distribution" in content


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
