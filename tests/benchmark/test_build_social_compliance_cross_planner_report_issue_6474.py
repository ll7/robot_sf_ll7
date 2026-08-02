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
