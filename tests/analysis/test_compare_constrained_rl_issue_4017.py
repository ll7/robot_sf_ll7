"""Tests for issue #4017 constrained-RL diagnostic comparison reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.analysis.compare_constrained_rl_issue_4017 import build_report_from_config

if TYPE_CHECKING:
    from pathlib import Path


def test_build_report_records_constraints_runtime_and_multiplier_trajectory(
    tmp_path: Path,
) -> None:
    """Synthetic smoke manifests produce the diagnostic comparison report shape."""

    baseline_manifest = tmp_path / "baseline_manifest.json"
    constrained_manifest = tmp_path / "constrained_manifest.json"
    constrained_trace = tmp_path / "constraint_trace.jsonl"
    output_json = tmp_path / "comparison_report.json"
    output_md = tmp_path / "comparison_report.md"
    _write_json(
        baseline_manifest,
        _manifest(
            policy_id="ppo_unconstrained_issue_4017_smoke",
            algorithm="ppo_unconstrained",
            constraints_enabled=False,
            constraints=[],
            runtime_seconds=1.25,
            trace_path=tmp_path / "empty_trace.jsonl",
        ),
    )
    _write_json(
        constrained_manifest,
        _manifest(
            policy_id="ppo_lagrangian_issue_4017_smoke",
            algorithm="ppo_lagrangian",
            constraints_enabled=True,
            constraints=[
                {
                    "name": "collision_any",
                    "source_key": "collision_any",
                    "budget_per_episode": 0.0,
                    "multiplier_init": 1.0,
                    "multiplier_lr": 0.05,
                    "multiplier_max": 50.0,
                    "normalize_by_episode_steps": False,
                },
                {
                    "name": "near_miss",
                    "source_key": "near_miss",
                    "budget_per_episode": 0.05,
                    "multiplier_init": 0.5,
                    "multiplier_lr": 0.02,
                    "multiplier_max": 20.0,
                    "normalize_by_episode_steps": False,
                },
            ],
            runtime_seconds=1.5,
            trace_path=constrained_trace,
        ),
    )
    constrained_trace.write_text(
        json.dumps(
            {
                "timesteps": 64,
                "episode_steps": 32,
                "costs": {"collision_any": 1.0, "near_miss": 0.2},
                "budgets": {"collision_any": 0.0, "near_miss": 0.05},
                "violations": {"collision_any": 1.0, "near_miss": 0.15},
                "multipliers_after_update": {"collision_any": 1.05, "near_miss": 0.503},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = _write_config(
        tmp_path,
        baseline_manifest=baseline_manifest,
        constrained_manifest=constrained_manifest,
        output_json=output_json,
        output_md=output_md,
    )

    report = build_report_from_config(config_path)

    assert report["status"] == "diagnostic_ready"
    assert report["paired_seed_count"] == 1
    assert report["fallback_or_degraded"] is False
    assert report["constraint_effect"]["benchmark_safety_claim"] is False
    assert report["constraint_effect"]["constrained_runtime_delta_seconds"] == pytest.approx(0.25)
    assert report["constraint_effect"]["budget_violation_constraints"] == [
        "collision_any",
        "near_miss",
    ]
    assert report["constraint_trace"]["record_count"] == 1
    assert report["constraint_trace"]["constraints"]["collision_any"]["latest_multiplier"] == 1.05
    assert json.loads(output_json.read_text(encoding="utf-8"))["status"] == "diagnostic_ready"
    assert "diagnostic smoke comparison only" in output_md.read_text(encoding="utf-8")


def test_report_marks_missing_runtime_and_empty_trace_as_blocked(tmp_path: Path) -> None:
    """Older manifests are accepted but explicit blockers prevent overclaiming."""

    baseline_manifest = tmp_path / "baseline_manifest.json"
    constrained_manifest = tmp_path / "constrained_manifest.json"
    constrained_trace = tmp_path / "constraint_trace.jsonl"
    constrained_trace.write_text("", encoding="utf-8")
    _write_json(
        baseline_manifest,
        _manifest(
            policy_id="ppo_unconstrained_issue_4017_smoke",
            algorithm="ppo_unconstrained",
            constraints_enabled=False,
            constraints=[],
            runtime_seconds=None,
            trace_path=tmp_path / "empty_trace.jsonl",
        ),
    )
    _write_json(
        constrained_manifest,
        _manifest(
            policy_id="ppo_lagrangian_issue_4017_smoke",
            algorithm="ppo_lagrangian",
            constraints_enabled=True,
            constraints=[
                {
                    "name": "collision_any",
                    "source_key": "collision_any",
                    "budget_per_episode": 0.0,
                    "multiplier_init": 1.0,
                    "multiplier_lr": 0.05,
                    "multiplier_max": 50.0,
                }
            ],
            runtime_seconds=None,
            trace_path=constrained_trace,
        ),
    )
    config_path = _write_config(
        tmp_path,
        baseline_manifest=baseline_manifest,
        constrained_manifest=constrained_manifest,
        output_json=tmp_path / "comparison_report.json",
        output_md=tmp_path / "comparison_report.md",
    )

    report = build_report_from_config(config_path)

    assert report["status"] == "diagnostic_blocked"
    assert "baseline manifest missing runtime_seconds" in report["blockers"]
    assert "constrained manifest missing runtime_seconds" in report["blockers"]
    assert "constrained trace has no completed episode multiplier records" in report["blockers"]


def test_missing_constraint_trace_fails_closed(tmp_path: Path) -> None:
    """A constrained manifest without its trace cannot be reported as evidence."""

    baseline_manifest = tmp_path / "baseline_manifest.json"
    constrained_manifest = tmp_path / "constrained_manifest.json"
    missing_trace = tmp_path / "missing_trace.jsonl"
    _write_json(
        baseline_manifest,
        _manifest(
            policy_id="ppo_unconstrained_issue_4017_smoke",
            algorithm="ppo_unconstrained",
            constraints_enabled=False,
            constraints=[],
            runtime_seconds=1.0,
            trace_path=tmp_path / "empty_trace.jsonl",
        ),
    )
    _write_json(
        constrained_manifest,
        _manifest(
            policy_id="ppo_lagrangian_issue_4017_smoke",
            algorithm="ppo_lagrangian",
            constraints_enabled=True,
            constraints=[],
            runtime_seconds=1.0,
            trace_path=missing_trace,
        ),
    )
    config_path = _write_config(
        tmp_path,
        baseline_manifest=baseline_manifest,
        constrained_manifest=constrained_manifest,
        output_json=tmp_path / "comparison_report.json",
        output_md=tmp_path / "comparison_report.md",
    )

    with pytest.raises(FileNotFoundError, match="constraint trace does not exist"):
        build_report_from_config(config_path)


def _manifest(
    *,
    policy_id: str,
    algorithm: str,
    constraints_enabled: bool,
    constraints: list[dict[str, object]],
    runtime_seconds: float | None,
    trace_path: Path,
) -> dict[str, object]:
    manifest = {
        "policy_id": policy_id,
        "algorithm": algorithm,
        "evidence_tier": "smoke",
        "claim_boundary": "diagnostic constrained-RL training smoke",
        "dry_run": False,
        "scenario_config": "configs/scenarios/sets/classic_cross_trap_subset.yaml",
        "seed": 4017,
        "total_timesteps": 256,
        "num_envs": 1,
        "device": "cpu",
        "constraints_enabled": constraints_enabled,
        "constraints": constraints,
        "constraint_trace_path": str(trace_path),
        "fallback_or_degraded": False,
    }
    if runtime_seconds is not None:
        manifest["runtime_seconds"] = runtime_seconds
    return manifest


def _write_config(
    tmp_path: Path,
    *,
    baseline_manifest: Path,
    constrained_manifest: Path,
    output_json: Path,
    output_md: Path,
) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "issue_4017.constrained_rl_diagnostic.v1",
                "issue": 4017,
                "evidence_tier": "diagnostic-only",
                "claim_boundary": "matched CPU smoke comparison only",
                "baseline_manifest": str(baseline_manifest),
                "constrained_manifest": str(constrained_manifest),
                "output_json": str(output_json),
                "output_markdown": str(output_md),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
