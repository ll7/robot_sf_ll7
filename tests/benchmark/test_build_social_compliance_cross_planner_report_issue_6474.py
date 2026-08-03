"""Tests for the preregistered issue #6474 social-compliance cross-planner report builder.

The script is a *preregistration*: its analysis plan is frozen before the nominal
campaign data exist.  These tests keep that freeze under CI, so protocol drift
(planner pairs, scenario suite, seed block, decision family, pinned config
digest) fails a check rather than silently changing what the campaign will be
allowed to claim.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/benchmark/build_social_compliance_cross_planner_report_issue_6474.py"
)


def _load_module() -> ModuleType:
    """Load the script module by path because scripts/benchmark is not a package."""
    spec = importlib.util.spec_from_file_location("issue6474_social_compliance_report", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


issue6474 = _load_module()


def test_self_test_recovers_the_known_answer_protocol() -> None:
    """The known-answer self-test is the script's proof surface; CI must run it."""

    assert issue6474.run_self_test() == 0


def test_frozen_protocol_matches_the_preregistration() -> None:
    """Planner pairs, scenarios, seeds, and the decision family stay frozen."""

    assert issue6474.FROZEN_PLANNERS == ("goal", "social_force", "orca")
    assert issue6474.FROZEN_PLANNER_PAIRS == (
        ("goal", "social_force"),
        ("goal", "orca"),
        ("social_force", "orca"),
    )
    assert issue6474.FROZEN_SCENARIOS == (
        "classic_head_on_corridor_medium",
        "classic_doorway_medium",
        "classic_group_crossing_medium",
        "classic_merging_medium",
        "classic_overtaking_medium",
        "classic_station_platform_medium",
    )
    assert issue6474.FROZEN_SEEDS == frozenset(range(111, 141))
    assert set(issue6474.FROZEN_SCENARIO_FAMILIES) == set(issue6474.FROZEN_SCENARIOS)
    # Three planner pairs x five metric families = the 15 Holm-controlled decisions.
    assert len(issue6474.METRIC_IDS) == 5
    decision_count = len(issue6474.FROZEN_PLANNER_PAIRS) * len(issue6474.METRIC_IDS)
    assert decision_count == 15


def test_frozen_config_digest_is_pinned() -> None:
    """The preregistered campaign config is pinned by digest, not by path alone."""

    digest = issue6474.FROZEN_CONFIG_SHA256
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    ("config_contents", "reason"),
    (
        (None, "absent config"),
        ("planners: [goal]\n", "config that drifted from the frozen digest"),
    ),
)
def test_cli_fails_closed_without_the_preregistered_config(
    tmp_path: Path,
    config_contents: str | None,
    reason: str,
) -> None:
    """A missing or drifted campaign config must exit 2, never produce a report."""

    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    config_path = tmp_path / "campaign.yaml"
    if config_contents is not None:
        config_path.write_text(config_contents, encoding="utf-8")
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--campaign-manifest",
            str(manifest_path),
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2, f"{reason} did not fail closed: {result.stderr}"
    assert "frozen SHA-256" in result.stderr
    assert not output_dir.exists(), "a fail-closed run must not write report artifacts"


def test_fallback_execution_rows_are_rejected() -> None:
    """Zero fallback/degraded rows is the nominal campaign contract."""

    row = issue6474._synthetic_row(
        "goal",
        issue6474.FROZEN_SCENARIOS[0],
        min(issue6474.FROZEN_SEEDS),
        {"comfort_exposure_person_s": 1.0},
        unavailable=(),
        execution_mode="fallback",
    )
    with pytest.raises(issue6474.AnalysisError):
        issue6474.validate_rows([row])


def _raw_episode(
    *,
    planner: str = "goal",
    scenario_id: str = "classic_head_on_corridor_medium",
    seed: int = 111,
    execution_mode: str = "native",
    adapter_active: bool = False,
) -> dict[str, object]:
    """Build the raw camera-ready episode shape used by the adapter tests."""

    metrics: dict[str, dict[str, object]] = {}
    for metric_id, (_family, units, denominator) in issue6474.METRIC_CONTRACT.items():
        metrics[metric_id] = {
            "id": metric_id,
            "status": "available",
            "support_count": 12,
            "denominator": denominator,
            "units": units,
            "value": 1.25,
        }
    return {
        "algo": planner,
        "scenario_id": scenario_id,
        "seed": seed,
        "status": "collision",
        "metrics": {
            "social_compliance": {
                "schema_version": issue6474.SOCIAL_COMPLIANCE_SCHEMA_VERSION,
                "claim_class": issue6474.SOCIAL_COMPLIANCE_CLAIM_CLASS,
                "metrics": metrics,
            },
        },
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {
                "execution_mode": execution_mode,
                "adapter_active": adapter_active,
            },
        },
    }


def _write_raw_input(
    tmp_path: Path,
    *,
    raw_rows: list[dict[str, object]],
    execution_mode: str = "native",
    benchmark_success: str = "true",
) -> tuple[Path, Path]:
    """Write a minimal manifest, campaign summary, and raw JSONL tree."""

    manifest_path = tmp_path / "campaign_manifest.json"
    manifest_path.write_text(json.dumps({"campaign_id": "test"}), encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "campaign_summary.json").write_text(
        json.dumps(
            {
                "planner_rows": [
                    {
                        "algo": "goal",
                        "execution_mode": execution_mode,
                        "readiness_status": execution_mode,
                        "availability_status": "available",
                        "benchmark_success": benchmark_success,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    episode_path = tmp_path / "runs" / "goal__differential_drive" / "episodes.jsonl"
    episode_path.parent.mkdir(parents=True)
    episode_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    return manifest_path, episode_path.parent.parent


def test_raw_jsonl_directory_is_normalized_with_campaign_status_context(tmp_path: Path) -> None:
    """Raw episode outcomes must be separated from campaign execution status."""

    manifest_path, runs_path = _write_raw_input(
        tmp_path,
        raw_rows=[
            _raw_episode(seed=111),
            _raw_episode(seed=112),
        ],
    )

    rows = issue6474.load_episode_rows(manifest_path, runs_path)

    assert [row["seed"] for row in rows] == [111, 112]
    assert rows[0]["execution_mode"] == "native"
    assert rows[0]["readiness_status"] == "native"
    assert rows[0]["benchmark_success"] is True
    assert rows[0]["execution_mode_consistent"] is True
    assert rows[0]["statuses"]["comfort_exposure_person_s"] == "available"
    assert rows[0]["values"]["comfort_exposure_person_s"] == 1.25


def test_raw_execution_mode_mismatch_is_fail_closed() -> None:
    """The adapter must not mark inconsistent raw metadata as benchmark-success."""

    raw = _raw_episode(execution_mode="adapter", adapter_active=True)
    context = {
        "goal": {
            "execution_mode": "native",
            "readiness_status": "native",
            "availability_status": "available",
            "benchmark_success": True,
        },
    }

    row = issue6474._flatten_raw_episode_row(raw, index=0, campaign_status=context)

    assert row["execution_mode_consistent"] is False
    with pytest.raises(issue6474.AnalysisError, match="benchmark-success campaign status"):
        issue6474.validate_rows([row])


def test_raw_jsonl_requires_summary_status_context(tmp_path: Path) -> None:
    """Raw records without campaign-level status evidence must fail closed."""

    raw_path = tmp_path / "episodes.jsonl"
    raw_path.write_text(json.dumps(_raw_episode()) + "\n", encoding="utf-8")

    with pytest.raises(issue6474.AnalysisError, match="requires campaign summary"):
        issue6474._load_rows_from_path(raw_path)


def test_malformed_jsonl_fails_closed(tmp_path: Path) -> None:
    """A malformed raw artifact must not produce a partial analysis."""

    raw_path = tmp_path / "episodes.jsonl"
    raw_path.write_text('{"algo": "goal"}\nnot-json\n', encoding="utf-8")

    with pytest.raises(issue6474.AnalysisError, match="not valid JSON"):
        issue6474._load_rows_from_path(raw_path)
