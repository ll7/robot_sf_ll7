"""Tests for the Social-Navigation-PyEnvs ORCA parity probe."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts.tools import probe_social_navigation_pyenvs_orca_parity as probe

if TYPE_CHECKING:
    from pathlib import Path


def _scenario(
    *,
    name: str,
    wrapper_mean: float,
    wrapper_max: float,
    oracle_mean: float,
    oracle_max: float,
    mismatch_steps: int,
) -> probe.ScenarioParitySummary:
    return probe.ScenarioParitySummary(
        name=name,
        scenario=name,
        seed=1000,
        trace_steps=5,
        wrapper_mean_xy_error=wrapper_mean,
        wrapper_max_xy_error=wrapper_max,
        oracle_mean_xy_error=oracle_mean,
        oracle_max_xy_error=oracle_max,
        heading_velocity_mismatch_steps=mismatch_steps,
        sample_rows=[
            {
                "t": 0.25,
                "upstream_action_xy": [0.3, -0.2],
                "wrapper_action_xy": [0.6, 0.0],
                "oracle_heading_action_xy": [0.3, -0.2],
                "robot_yaw": 0.0,
                "velocity_heading_rad": -0.7,
            }
        ],
    )


def test_determine_verdict_flags_material_contract_mismatch() -> None:
    """High wrapper error with near-perfect oracle parity should be treated as mismatch."""
    verdict, root_cause, projection_role = probe._determine_verdict(
        [
            _scenario(
                name="parallel_traffic_orca",
                wrapper_mean=0.18,
                wrapper_max=0.31,
                oracle_mean=0.0,
                oracle_max=0.0,
                mismatch_steps=5,
            )
        ]
    )

    assert verdict == "adapter has material contract mismatch"
    assert "full planar self velocity" in root_cause
    assert "before projection" in projection_role


def test_determine_verdict_accepts_clean_source_parity() -> None:
    """Tiny wrapper error across scenarios should report source-faithful parity."""
    verdict, root_cause, projection_role = probe._determine_verdict(
        [
            _scenario(
                name="circular_crossing_hsfm_new_guo",
                wrapper_mean=0.0,
                wrapper_max=0.0,
                oracle_mean=0.0,
                oracle_max=0.0,
                mismatch_steps=0,
            ),
            _scenario(
                name="parallel_traffic_orca",
                wrapper_mean=0.0,
                wrapper_max=0.0,
                oracle_mean=0.0,
                oracle_max=0.0,
                mismatch_steps=0,
            ),
        ]
    )

    assert verdict == "adapter appears source-faithful but benchmark-misaligned"
    assert "Raw upstream ActionXY traces match" in root_cause
    assert "downstream unicycle execution" in projection_role


def test_summarize_scenario_counts_heading_velocity_mismatch_steps() -> None:
    """Mismatch counting should ignore stationary rows and count moving heading divergence."""
    spec = {"name": "demo", "scenario": "parallel_traffic", "seed": 7}
    payload = {
        "rows": [
            {
                "robot_speed": 0.0,
                "robot_yaw": 0.0,
                "velocity_heading_rad": 1.0,
                "upstream_action_xy": [0.0, 1.0],
                "wrapper_action_xy": [0.0, 1.0],
                "oracle_heading_action_xy": [0.0, 1.0],
            },
            {
                "robot_speed": 0.5,
                "robot_yaw": 0.0,
                "velocity_heading_rad": 0.4,
                "upstream_action_xy": [0.4, -0.2],
                "wrapper_action_xy": [0.6, 0.0],
                "oracle_heading_action_xy": [0.4, -0.2],
            },
        ]
    }

    summary = probe._summarize_scenario(spec, payload)

    assert summary.trace_steps == 2
    assert summary.heading_velocity_mismatch_steps == 1
    assert summary.wrapper_mean_xy_error > 0.0
    assert summary.oracle_mean_xy_error == 0.0


def test_render_markdown_records_parallel_traffic_failure_mode(tmp_path: Path) -> None:
    """Markdown should call out the self-velocity mismatch, not just poor performance."""
    report = probe.ParityReport(
        issue=649,
        repo_root=str(tmp_path / "repo"),
        repo_remote_url="https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
        verdict="adapter has material contract mismatch",
        root_cause=(
            "Robot SF SocNav observations expose heading plus scalar speed, while the upstream "
            "ORCA path consumes full planar self velocity."
        ),
        projection_role=(
            "The ActionXY->unicycle_vw projection is still a second mismatch layer, but the raw "
            "ActionXY parity already fails before projection on nontrivial upstream traces."
        ),
        scenarios=[
            _scenario(
                name="parallel_traffic_orca",
                wrapper_mean=0.18,
                wrapper_max=0.31,
                oracle_mean=0.0,
                oracle_max=0.0,
                mismatch_steps=5,
            )
        ],
    )

    markdown = probe._render_markdown(report)

    assert "Verdict: `adapter has material contract mismatch`" in markdown
    assert "parallel_traffic_orca" in markdown
    assert "wrapper mean ActionXY error" in markdown
    assert "self-velocity contract" in markdown


def test_render_markdown_updates_interpretation_for_clean_parity(tmp_path: Path) -> None:
    """Clean parity verdicts should not keep the old mismatch interpretation text."""
    report = probe.ParityReport(
        issue=649,
        repo_root=str(tmp_path / "repo"),
        repo_remote_url="https://github.com/TommasoVandermeer/Social-Navigation-PyEnvs",
        verdict="adapter appears source-faithful but benchmark-misaligned",
        root_cause="Raw upstream ActionXY traces match across the tested upstream scenarios.",
        projection_role=(
            "Remaining performance differences would then be explained primarily by scenario "
            "mismatch and downstream unicycle execution."
        ),
        scenarios=[
            _scenario(
                name="parallel_traffic_orca",
                wrapper_mean=0.0,
                wrapper_max=0.0,
                oracle_mean=0.0,
                oracle_max=0.0,
                mismatch_steps=6,
            )
        ],
    )

    markdown = probe._render_markdown(report)

    assert "now also matches upstream raw `ActionXY`" in markdown
    assert "cannot reconstruct the same self velocity" not in markdown
