"""Tests for the issue #5578 campaign manifest compiler and disjoint-seed preflight.

These tests prove the #6101 deliverables:
- the manifest materializes exactly 2,160 registered identities with every frozen
  runtime value and rejects duplicate / missing / drifted cells;
- the check-only CLI validates the plan with no execution side effects;
- the activation rule frozen by #6100 is evaluated correctly;
- the disjoint-seed preflight exercises the real drive/action binding and satisfies
  the activation gate;
- the documented full-run surface fails closed (registered execution is not
  authorized in this issue);
- the synthesis adapter connects file-backed campaign rows to the reviewed
  synthesizer and fails closed.

No registered seed 111-140 is executed by these tests.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.benchmark.issue_5578_speed_tier_synthesis import (
    DECLARED_PLANNERS,
    DECLARED_SCENARIOS,
    DECLARED_SEEDS,
    MIN_ACTIVATION_FRACTION_ABOVE_2_0,
    MIN_ACTIVATION_PEAK_SPEED,
    NOMINAL_TIER_ID,
    NON_NOMINAL_TIERS,
    TIER_ACTUATION_ENVELOPES,
    TIER_CAPS_M_S,
)
from scripts.benchmark import run_issue_5578_speed_tier_campaign as campaign
from scripts.benchmark.run_issue_5578_speed_tier_campaign import (
    AMENDMENT_ISSUE,
    CLAIM_BOUNDARY,
    EXPECTED_CELL_COUNT,
    NOT_EVIDENCE_BANNER,
    PREFLIGHT_SCENARIO,
    PREFLIGHT_SEEDS,
    THIS_ISSUE,
    CampaignManifestError,
    FullRunBlockedError,
    compile_campaign_manifest,
    evaluate_activation_rule,
    main,
    manifest_to_dict,
    run_activation_preflight,
    synthesize_from_cell_summaries,
)

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_SCHEMA_VERSION_MARKER = campaign.MANIFEST_SCHEMA_VERSION

DISJOINT_SEEDS = (211, 212)


def _manifest() -> Any:
    return compile_campaign_manifest()


def _successful_runner_summary(written: int) -> dict[str, Any]:
    """Return the fail-closed runner summary required by authorized execution."""
    return {
        "written": written,
        "failed_jobs": 0,
        "skipped_jobs": 0,
        "failures": [],
        "benchmark_availability": {
            "availability_status": "available",
            "benchmark_success": True,
        },
        "provenance": {"result_manifest_status": "available"},
    }


def _fake_episode_row(
    scenario: dict[str, Any],
    seed: int,
    *,
    algo: str,
    cap: float,
    status: str = "ok",
) -> dict[str, Any]:
    """Build one canonical fake map-runner row for authorized-lane tests."""
    planner_mode = "adapter" if algo == "orca" else "native"
    return {
        "episode_id": f"{scenario['name']}--{seed}",
        "scenario_id": scenario["name"],
        "seed": seed,
        "horizon": 600,
        "algorithm_metadata": {
            "status": status,
            "canonical_algorithm": algo,
            "planner_kinematics": {"execution_mode": planner_mode},
            "simulation_step_trace": {
                "dt": 0.1,
                "steps": [
                    {
                        "robot": {"velocity": [cap, 0.0]},
                        "planner": {"selected_action": {"linear_velocity": cap}},
                    },
                    {
                        "robot": {"velocity": [cap, 0.0]},
                        "planner": {"selected_action": {"linear_velocity": cap}},
                    },
                ],
            },
        },
        "metrics": {
            "success": True,
            "total_collision_count": 0,
            "collisions": 0,
            "ped_collision_count": 0,
            "obstacle_collision_count": 0,
            "agent_collision_count": 0,
            "near_misses": 0,
            "time_to_goal_norm": 0.5,
            "socnavbench_path_length": 1.0,
            "mean_clearance": 1.0,
            "min_clearance": 0.5,
        },
        "interaction_exposure": {
            "interaction_exposure_share": 0.0,
            "interaction_exposure_denominator_steps": 2,
        },
        "config_hash": "a" * 16,
        "git_hash": "a" * 40,
        "result_provenance": {
            "schema_version": "benchmark_row_provenance.v1",
            "scenario_id": scenario["name"],
            "seed": seed,
            "config_hash": "a" * 16,
            "repo_commit": "a" * 40,
            "simulator_settings": {"horizon": 600, "dt": 0.1},
            "postprocessing": [
                {"step": "compute_all_metrics", "status": "completed"},
                {"step": "post_process_metrics", "status": "completed"},
            ],
        },
    }


def test_manifest_has_exactly_2160_unique_registered_identities() -> None:
    """The manifest materializes the exact 6x3x4x30 frozen cross."""
    manifest = _manifest()
    assert len(manifest.identities) == EXPECTED_CELL_COUNT
    keys = [row["identity_key"] for row in manifest.identities]
    assert len(set(keys)) == len(keys)
    assert manifest.expected_cell_count == EXPECTED_CELL_COUNT
    assert manifest.manifest_hash
    assert len(manifest.scenarios) == 6
    assert len(manifest.speed_tiers) == 3
    assert len(manifest.planners) == 4
    assert len(manifest.seeds) == 30


def test_manifest_freezes_scenarios_tiers_planners_seeds() -> None:
    """Frozen dimensions match the #6100 preregistration exactly."""
    manifest = _manifest()
    assert {s.scenario_id for s in manifest.scenarios} == set(DECLARED_SCENARIOS)
    assert {p.planner_id for p in manifest.planners} == set(DECLARED_PLANNERS)
    assert set(manifest.seeds) == set(DECLARED_SEEDS)
    assert [t.tier_id for t in manifest.speed_tiers] == [
        NOMINAL_TIER_ID,
        *NON_NOMINAL_TIERS,
    ]
    assert {t.tier_id: t.cap_m_s for t in manifest.speed_tiers} == dict(TIER_CAPS_M_S)
    assert all(t.drive_model == "bicycle_drive" for t in manifest.speed_tiers)


def test_manifest_carries_full_frozen_runtime_values_per_identity() -> None:
    """Every identity carries drive model, accel/decel, stopping envelope, and contracts."""
    manifest = _manifest()
    tier_by_id = {t.tier_id: t for t in manifest.speed_tiers}
    for row in manifest.identities:
        tier = tier_by_id[row["speed_tier_id"]]
        assert row["speed_cap_m_s"] == tier.cap_m_s
        assert row["drive_model"] == "bicycle_drive"
        assert row["max_accel_m_s2"] == tier.max_accel_m_s2
        assert row["max_decel_m_s2"] == tier.max_decel_m_s2
        assert row["stopping_distance_envelope_m"] == tier.stopping_distance_envelope_m
        assert row["horizon_steps"] == 600
        assert row["dt_seconds"] == pytest.approx(0.1)
        assert row["execution_mode"] == "native"
        assert row["registered"] is True
        assert row["runtime_variant_key"] == tier.runtime_variant_key
        assert "resolved_actuation_envelope" in row
        assert "planner_command_contract" in row
        assert "environment_action_contract" in row


def test_manifest_resolved_envelopes_match_frozen_actuation_contract() -> None:
    """Each tier's resolved #4976 envelope matches the synthesizer's frozen values."""
    manifest = _manifest()
    for tier in manifest.speed_tiers:
        frozen = TIER_ACTUATION_ENVELOPES[tier.tier_id]
        for key, expected in frozen.items():
            actual = tier.resolved_actuation_envelope[key]
            if isinstance(expected, str):
                assert actual == expected
            else:
                assert math.isclose(float(actual), expected, abs_tol=1e-9), (
                    f"{tier.tier_id}.{key}: expected {expected}, got {actual}"
                )
        # Stopping distance must equal v^2 / (2*decel).
        cap = tier.cap_m_s
        decel = tier.max_decel_m_s2
        assert tier.stopping_distance_envelope_m == pytest.approx(cap**2 / (2 * decel))


def test_manifest_top_tier_is_amended_4_0_not_4_2() -> None:
    """The top tier is the supported 4.0 m/s variant amended by #6100, not 4.2 m/s."""
    manifest = _manifest()
    top = manifest.speed_tiers[-1]
    assert top.tier_id == "cap_4_0"
    assert top.cap_m_s == 4.0
    assert top.runtime_variant_key == "bicycle_4_0_mps_micromobility"


def test_manifest_rejects_duplicate_identities() -> None:
    """A duplicate identity key must fail closed."""
    manifest = _manifest()
    identities = [dict(row) for row in manifest.identities]
    identities.append(dict(identities[0]))
    with pytest.raises(CampaignManifestError, match="identity count is not 2160"):
        # Re-run the grid validation directly to prove duplicate rejection.
        campaign._validate_identity_grid(
            identities,
            manifest.scenarios,
            manifest.speed_tiers,
            manifest.planners,
            manifest.seeds,
        )


def test_manifest_rejects_missing_cells() -> None:
    """A missing cell must fail closed."""
    manifest = _manifest()
    identities = [dict(row) for row in manifest.identities][:-1]
    with pytest.raises(CampaignManifestError, match="identity count is not 2160"):
        campaign._validate_identity_grid(
            identities,
            manifest.scenarios,
            manifest.speed_tiers,
            manifest.planners,
            manifest.seeds,
        )


def test_manifest_runtime_resolution_documents_canonical_surfaces() -> None:
    """The manifest records the canonical runtime surfaces it binds to."""
    manifest = _manifest()
    resolution = manifest.runtime_resolution
    assert resolution["runtime_converter"].endswith("::_env_action")
    assert resolution["speed_cap_reader"].endswith("::_robot_speed_cap")
    assert resolution["angular_cap_reader"].endswith("::_robot_angular_cap")
    assert (
        resolution["native_action_space"]
        == "robot_sf.robot.bicycle_drive.BicycleDriveRobot.action_space"
    )
    assert set(resolution["resolved_actuation_envelopes_by_tier"]) == {
        NOMINAL_TIER_ID,
        *NON_NOMINAL_TIERS,
    }


def test_manifest_to_dict_is_json_serializable_and_complete() -> None:
    """The serialized manifest round-trips through JSON with all sections."""
    payload = manifest_to_dict(_manifest())
    text = json.dumps(payload, sort_keys=True)
    assert MANIFEST_SCHEMA_VERSION_MARKER in text
    decoded = json.loads(text)
    assert decoded["schema_version"] == campaign.MANIFEST_SCHEMA_VERSION
    assert decoded["issue"] == 5578
    assert decoded["amendment_issue"] == AMENDMENT_ISSUE
    assert decoded["this_issue"] == THIS_ISSUE
    assert decoded["claim_boundary"] == CLAIM_BOUNDARY
    assert len(decoded["identities"]) == EXPECTED_CELL_COUNT
    assert decoded["frozen_contract"]["expected_cell_count"] == EXPECTED_CELL_COUNT


def test_activation_rule_nominal_tier_is_reference_not_treated() -> None:
    """The 2.0 m/s nominal tier is the reference axis level, not a treated intervention."""
    gate = evaluate_activation_rule(0.0, 2.0, tier_id=NOMINAL_TIER_ID)
    assert gate["activated"] is True
    assert gate["applicability"] == "nominal_reference_not_a_treated_intervention"


@pytest.mark.parametrize("tier_id", list(NON_NOMINAL_TIERS))
def test_activation_rule_non_nominal_tier_activates_on_fraction(tier_id: str) -> None:
    """Non-nominal tiers activate when fraction above 2.0 m/s meets the threshold."""
    gate = evaluate_activation_rule(0.06, 2.0, tier_id=tier_id)
    assert gate["activated"] is True
    assert gate["fraction_above_2_0_mps_passes"] is True


@pytest.mark.parametrize("tier_id", list(NON_NOMINAL_TIERS))
def test_activation_rule_non_nominal_tier_activates_on_peak(tier_id: str) -> None:
    """Non-nominal tiers activate when realized peak speed exceeds 2.2 m/s."""
    gate = evaluate_activation_rule(0.0, 2.5, tier_id=tier_id)
    assert gate["activated"] is True
    assert gate["peak_speed_passes"] is True


def test_activation_rule_cap_inactive_when_below_thresholds() -> None:
    """A non-nominal tier below both thresholds is cap-inactive."""
    gate = evaluate_activation_rule(0.01, 1.9, tier_id="cap_3_0")
    assert gate["activated"] is False
    assert gate["fraction_above_2_0_mps_passes"] is False
    assert gate["peak_speed_passes"] is False


def test_activation_rule_thresholds_match_frozen_contract() -> None:
    """The activation thresholds are exactly those frozen by #6100."""
    gate = evaluate_activation_rule(0.0, 0.0, tier_id="cap_4_0")
    assert gate["fraction_above_2_0_mps_threshold"] == MIN_ACTIVATION_FRACTION_ABOVE_2_0
    assert gate["realized_speed_peak_m_s_threshold"] == MIN_ACTIVATION_PEAK_SPEED


def test_check_only_cli_writes_manifest_with_no_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check-only CLI validates the plan and writes only the declared manifest file."""
    out = tmp_path / "manifest.json"
    rc = main(["--check-only", "--manifest-out", str(out)])
    assert rc == 0
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert len(payload["identities"]) == EXPECTED_CELL_COUNT
    captured = capsys.readouterr().out
    assert "side_effects: none" in captured
    assert "2160" in captured


def test_check_only_cli_without_manifest_out_emits_compact_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Check-only without an output path still validates and prints the identity count."""
    rc = main(["--check-only"])
    assert rc == 0
    assert "2160" in capsys.readouterr().out


def test_full_run_cli_fails_closed(capsys: pytest.CaptureFixture[str]) -> None:
    """The documented full-run surface refuses to execute registered episodes here."""
    with pytest.raises(FullRunBlockedError):
        main(["--full-run", "--cell-summaries-out", "/tmp/issue-5578-cells.jsonl"])
    output = capsys.readouterr().out
    assert "not authorized" in output.lower()
    assert "--cell-summaries-out /tmp/issue-5578-cells.jsonl" in output


def test_authorized_full_run_requires_exact_operational_issue() -> None:
    """Registered execution cannot start without the explicit #6102 flag."""
    with pytest.raises(campaign.CampaignAuthorizationError, match="6102"):
        main(["--authorized-full-run"])


def test_authorized_runtime_preflight_requires_exact_operational_issue() -> None:
    """Runtime binding checks cannot start without the explicit #6102 flag."""
    with pytest.raises(campaign.CampaignAuthorizationError, match="6102"):
        main(["--authorized-runtime-preflight"])


def test_execution_disposition_allows_expected_adapter_but_rejects_fallback() -> None:
    """Planner command adapters remain native rows; fallback/degraded states do not."""
    adapter_record = {
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "adapter"},
        }
    }
    assert campaign._campaign_execution_disposition(adapter_record) == ("native", None)

    fallback_record = {"algorithm_metadata": {"status": "fallback"}}
    assert campaign._campaign_execution_disposition(fallback_record)[0] == "degraded"

    degraded_record = {
        "algorithm_metadata": {
            "status": "ok",
            "foresight_prediction": {"evidence_eligible": False},
        }
    }
    assert campaign._campaign_execution_disposition(degraded_record)[0] == "degraded"


def test_execution_disposition_requires_explicit_planner_execution_mode() -> None:
    """A healthy status without a native/adapter mode must not become campaign evidence."""
    missing_mode = {"algorithm_metadata": {"status": "ok"}}
    unknown_mode = {
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "unknown"},
        }
    }
    assert campaign._campaign_execution_disposition(missing_mode)[0] == "failed"
    assert campaign._campaign_execution_disposition(unknown_mode)[0] == "failed"


def test_episode_provenance_requires_canonical_row_contract() -> None:
    """Missing or fabricated row provenance must fail before cell conversion."""
    with pytest.raises(campaign.AuthorizedCampaignError, match="result_provenance"):
        campaign._validate_episode_provenance(
            {},
            scenario_id=PREFLIGHT_SCENARIO,
            seed=211,
            horizon_steps=600,
            dt_seconds=0.1,
            expected_repo_commit="a" * 40,
        )

    malformed = {
        "result_provenance": {
            "schema_version": "benchmark_row_provenance.v1",
            "scenario_id": PREFLIGHT_SCENARIO,
            "seed": 211,
            "config_hash": "not-a-hash",
            "repo_commit": "not-a-commit",
        }
    }
    with pytest.raises(campaign.AuthorizedCampaignError, match="config_hash"):
        campaign._validate_episode_provenance(
            malformed,
            scenario_id=PREFLIGHT_SCENARIO,
            seed=211,
            horizon_steps=600,
            dt_seconds=0.1,
            expected_repo_commit="a" * 40,
        )


def test_episode_provenance_requires_the_authorized_execution_commit() -> None:
    """Rows from another revision cannot be resumed into a registered campaign."""
    record = {
        "result_provenance": {
            "schema_version": "benchmark_row_provenance.v1",
            "scenario_id": PREFLIGHT_SCENARIO,
            "seed": 211,
            "config_hash": "a" * 16,
            "repo_commit": "a" * 40,
            "simulator_settings": {"horizon": 600, "dt": 0.1},
            "postprocessing": [
                {"step": "compute_all_metrics", "status": "completed"},
                {"step": "post_process_metrics", "status": "completed"},
            ],
        }
    }
    with pytest.raises(campaign.AuthorizedCampaignError, match="does not match"):
        campaign._validate_episode_provenance(
            record,
            scenario_id=PREFLIGHT_SCENARIO,
            seed=211,
            horizon_steps=600,
            dt_seconds=0.1,
            expected_repo_commit="b" * 40,
        )


def test_runner_summary_rejects_failed_jobs_before_reading_rows() -> None:
    """A runner-level failure cannot be hidden by a later row-count check."""
    with pytest.raises(campaign.AuthorizedCampaignError, match="failed jobs"):
        campaign._validate_runner_summary(
            {"failed_jobs": 1, "written": EXPECTED_CELL_COUNT},
            batch_name="cap_2_0_nominal__orca",
            expected_count=EXPECTED_CELL_COUNT,
            resume=False,
        )


@pytest.mark.parametrize(
    ("summary", "match"),
    [
        (
            {"written": 1, "failed_jobs": 0, "skipped_jobs": 0, "failures": []},
            "benchmark_availability",
        ),
        (
            {
                "written": 1,
                "failed_jobs": 0,
                "skipped_jobs": 0,
                "failures": [],
                "benchmark_availability": {
                    "availability_status": "available",
                    "benchmark_success": True,
                },
            },
            "provenance",
        ),
        (
            {
                **_successful_runner_summary(1),
                "provenance": {"result_manifest_status": "error"},
            },
            "not available",
        ),
    ],
)
def test_runner_summary_requires_availability_and_provenance(
    summary: dict[str, Any],
    match: str,
) -> None:
    """Malformed runner summaries cannot cross the authorized campaign boundary."""
    with pytest.raises(campaign.AuthorizedCampaignError, match=match):
        campaign._validate_runner_summary(
            summary,
            batch_name="cap_2_0_nominal__orca",
            expected_count=1,
            resume=False,
        )


def test_authorized_output_resume_requires_matching_descriptor(tmp_path: Path) -> None:
    """Resume accepts owned partial batches and rejects orphaned batch artifacts."""
    manifest = _manifest()
    raw_root = tmp_path / "owned"
    descriptor = campaign._prepare_authorized_output_root(
        raw_root,
        manifest,
        resume=False,
    )
    (raw_root / "partial.jsonl").write_text("{}\n", encoding="utf-8")
    (raw_root / "partial.summary.json").write_text("{}\n", encoding="utf-8")

    assert campaign._prepare_authorized_output_root(raw_root, manifest, resume=True) == descriptor

    orphan_root = tmp_path / "orphan"
    orphan_root.mkdir()
    (orphan_root / "partial.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(campaign.AuthorizedCampaignError, match="matching descriptor"):
        campaign._prepare_authorized_output_root(orphan_root, manifest, resume=True)


def test_runtime_preflight_requires_clean_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operational preflight cannot attest rows from uncommitted source state."""
    monkeypatch.setattr(
        campaign,
        "_git_provenance",
        lambda: {
            "git_head": "a" * 40,
            "git_worktree_dirty": True,
            "git_status_short": [" M local-change.py"],
        },
    )

    with pytest.raises(campaign.AuthorizedCampaignError, match="clean git worktree"):
        campaign.run_authorized_runtime_preflight(
            _manifest(),
            output_root=tmp_path / "preflight",
            authorization_issue=6102,
        )


def test_runtime_preflight_records_unexpected_runner_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected runner errors leave a terminal operational report."""
    monkeypatch.setattr(
        campaign,
        "_git_provenance",
        lambda: {
            "git_head": "a" * 40,
            "git_worktree_dirty": False,
            "git_status_short": [],
        },
    )
    monkeypatch.setattr(
        campaign,
        "_authorized_campaign_scenarios",
        lambda _manifest: {PREFLIGHT_SCENARIO: {"name": PREFLIGHT_SCENARIO}},
    )

    def fail_runner(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(campaign, "_run_native_batch", fail_runner)
    output_root = tmp_path / "preflight"
    with pytest.raises(campaign.AuthorizedCampaignError, match="runner exploded"):
        campaign.run_authorized_runtime_preflight(
            _manifest(),
            output_root=output_root,
            authorization_issue=6102,
        )

    report = json.loads((output_root / "runtime_preflight_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["error"] == "RuntimeError: runner exploded"


def test_authorized_campaign_fake_runner_covers_exact_2160_native_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The authorized lane enforces the exact grid before synthesis."""

    def fake_run_native_batch(
        scenarios: list[dict[str, Any]],
        out_path: Path,
        *,
        algo: str,
        algo_config_path: str | None,
        horizon_steps: int,
        dt_seconds: float,
        resume: bool,
    ) -> dict[str, Any]:
        del algo_config_path, resume
        assert horizon_steps == 600
        assert dt_seconds == 0.1
        cap = float(scenarios[0]["robot_config"]["max_velocity"])
        rows = [
            _fake_episode_row(scenario, seed, algo=algo, cap=cap)
            for scenario in scenarios
            for seed in scenario["seeds"]
        ]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return _successful_runner_summary(len(rows))

    monkeypatch.setattr(campaign, "_run_native_batch", fake_run_native_batch)
    monkeypatch.setattr(
        campaign,
        "_git_provenance",
        lambda: {"git_head": "a" * 40, "git_worktree_dirty": False, "git_status_short": []},
    )
    manifest = _manifest()
    report = campaign.execute_authorized_campaign(
        manifest,
        raw_root=tmp_path / "raw",
        cell_summaries_path=tmp_path / "cells.jsonl",
        synthesis_path=tmp_path / "synthesis.json",
        authorization_issue=6102,
    )
    assert report["status"] == "complete_native"
    assert report["native_cell_count"] == EXPECTED_CELL_COUNT
    assert report["git_provenance"]["git_head"]
    assert report["command_environment_manifest"]["runner"] == (
        "robot_sf.benchmark.map_runner.run_map_batch"
    )
    cells = (tmp_path / "cells.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(cells) == EXPECTED_CELL_COUNT
    assert all(json.loads(line)["execution_mode"] == "native" for line in cells)


def test_authorized_campaign_rejection_quarantines_non_native_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fallback/degraded row rejects the run without polluting the canonical artifact.

    The canonical cell-summaries path (consumed by ``--synthesize``) must never be written
    when the grid contains any non-native row. Rejected rows are quarantined to a
    distinctly-named ``.rejected.jsonl`` path for diagnosis, and the report records both
    the rejection and the quarantine path.
    """
    degrade_first = {"active": True}

    def fake_run_native_batch(
        scenarios: list[dict[str, Any]],
        out_path: Path,
        *,
        algo: str,
        algo_config_path: str | None,
        horizon_steps: int,
        dt_seconds: float,
        resume: bool,
    ) -> dict[str, Any]:
        del algo_config_path, resume
        assert horizon_steps == 600
        assert dt_seconds == 0.1
        cap = float(scenarios[0]["robot_config"]["max_velocity"])
        planner_mode = "adapter" if algo == "orca" else "native"
        rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            for seed in scenario["seeds"]:
                # The first emitted row is a fallback; the rest are native. A single
                # non-native row is enough to fail the whole campaign closed.
                status = "fallback" if degrade_first["active"] else "ok"
                degrade_first["active"] = False
                rows.append(
                    {
                        "episode_id": f"{scenario['name']}--{seed}",
                        "scenario_id": scenario["name"],
                        "seed": seed,
                        "horizon": 600,
                        "algorithm_metadata": {
                            "status": status,
                            "canonical_algorithm": algo,
                            "planner_kinematics": {"execution_mode": planner_mode},
                            "simulation_step_trace": {
                                "dt": 0.1,
                                "steps": [
                                    {
                                        "robot": {"velocity": [cap, 0.0]},
                                        "planner": {"selected_action": {"linear_velocity": cap}},
                                    },
                                    {
                                        "robot": {"velocity": [cap, 0.0]},
                                        "planner": {"selected_action": {"linear_velocity": cap}},
                                    },
                                ],
                            },
                        },
                        "metrics": {
                            "success": True,
                            "total_collision_count": 0,
                            "collisions": 0,
                            "ped_collision_count": 0,
                            "obstacle_collision_count": 0,
                            "agent_collision_count": 0,
                            "near_misses": 0,
                            "time_to_goal_norm": 0.5,
                            "socnavbench_path_length": 1.0,
                            "mean_clearance": 1.0,
                            "min_clearance": 0.5,
                        },
                        "interaction_exposure": {
                            "interaction_exposure_share": 0.0,
                            "interaction_exposure_denominator_steps": 2,
                        },
                        "config_hash": "a" * 16,
                        "git_hash": "a" * 40,
                        "result_provenance": {
                            "schema_version": "benchmark_row_provenance.v1",
                            "scenario_id": scenario["name"],
                            "seed": seed,
                            "config_hash": "a" * 16,
                            "repo_commit": "a" * 40,
                            "simulator_settings": {"horizon": 600, "dt": 0.1},
                            "postprocessing": [
                                {"step": "compute_all_metrics", "status": "completed"},
                                {"step": "post_process_metrics", "status": "completed"},
                            ],
                        },
                    }
                )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return _successful_runner_summary(len(rows))

    monkeypatch.setattr(campaign, "_run_native_batch", fake_run_native_batch)
    monkeypatch.setattr(
        campaign,
        "_git_provenance",
        lambda: {"git_head": "a" * 40, "git_worktree_dirty": False, "git_status_short": []},
    )
    manifest = _manifest()
    cells = tmp_path / "cells.jsonl"
    report_path = tmp_path / "report.json"
    with pytest.raises(campaign.AuthorizedCampaignError, match="rejected"):
        campaign.execute_authorized_campaign(
            manifest,
            raw_root=tmp_path / "raw",
            cell_summaries_path=cells,
            synthesis_path=tmp_path / "synthesis.json",
            report_path=report_path,
            authorization_issue=6102,
        )

    # Canonical path consumed by --synthesize must not exist for a rejected run.
    assert not cells.exists()
    # Rejected rows are quarantined under a distinct name and recorded in the report.
    quarantine = cells.with_suffix(".rejected.jsonl")
    assert quarantine.exists()
    quarantined = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    assert any(row["execution_mode"] != "native" for row in quarantined)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "rejected_non_native_rows"
    assert report["excluded_cell_count"] >= 1
    assert report["rejected_cell_summaries_path"] == campaign._repo_rel(quarantine)


def test_authorized_full_run_defaults_anchor_to_module_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default cell/synthesis paths anchor to module constants, not ``--raw-root``'s parent.

    A custom ``--raw-root`` (for example a bare ``raw``) must not drop
    ``cell_summaries.jsonl`` / ``synthesis.json`` into the repository root; the canonical
    module paths remain the defaults and explicit ``--cell-summaries-out`` /
    ``--synthesis-out`` overrides still win.
    """
    captured: dict[str, str] = {}

    def fake_execute(
        manifest: Any,
        *,
        raw_root: Any,
        cell_summaries_path: Any,
        synthesis_path: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del manifest, kwargs
        captured["raw_root"] = str(raw_root)
        captured["cell_summaries_path"] = str(cell_summaries_path)
        captured["synthesis_path"] = str(synthesis_path)
        return {
            "status": "complete_native",
            "native_cell_count": EXPECTED_CELL_COUNT,
            "manifest_hash": "test",
            "synthesis_path": str(synthesis_path),
            "report_path": "output/issue_5578_robot_speed_tier_sweep/report.json",
        }

    monkeypatch.setattr(campaign, "execute_authorized_campaign", fake_execute)
    rc = main(
        [
            "--authorized-full-run",
            "--authorization-issue",
            "6102",
            "--raw-root",
            "raw",
            "--json",
        ]
    )
    assert rc == 0
    # The bare ``raw`` --raw-root is honored for episodes but must not relocate the
    # canonical cell-summaries / synthesis artifacts into the repository root.
    assert captured["raw_root"] == "raw"
    assert captured["cell_summaries_path"] == campaign.DEFAULT_CELL_SUMMARY_PATH
    assert captured["synthesis_path"] == campaign.DEFAULT_SYNTHESIS_PATH


def test_synthesize_adapter_connects_to_reviewed_synthesizer(tmp_path: Path) -> None:
    """File-backed rows flow through the adapter into the reviewed synthesizer."""
    rows = _smoke_rows()
    cells = tmp_path / "cells.jsonl"
    cells.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = synthesize_from_cell_summaries(
        cells,
        output_path=tmp_path / "synthesis.json",
        declared_scenarios={PREFLIGHT_SCENARIO},
        declared_planners={"orca"},
        declared_seeds={111},
    )
    assert report["evidence_status"] == "smoke_or_incomplete_not_benchmark_evidence"
    assert report["native_cell_count"] == 3
    assert (tmp_path / "synthesis.json").is_file()


def test_synthesize_adapter_rejects_empty_file(tmp_path: Path) -> None:
    """An empty cell-summaries file fails closed."""
    cells = tmp_path / "empty.jsonl"
    cells.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        synthesize_from_cell_summaries(cells)


def test_synthesize_adapter_rejects_undeclared_seed(tmp_path: Path) -> None:
    """A row with an undeclared seed is rejected by the fail-closed synthesizer."""
    rows = _smoke_rows(seed=99999)
    cells = tmp_path / "cells.jsonl"
    cells.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared seed"):
        synthesize_from_cell_summaries(
            cells,
            declared_scenarios={PREFLIGHT_SCENARIO},
            declared_planners={"orca"},
            declared_seeds={111},
        )


def test_preflight_rejects_overlapping_registered_seeds() -> None:
    """The preflight must never use a registered seed from the 111-140 block."""
    manifest = _manifest()
    with pytest.raises(campaign.PreflightActivationError, match="overlap"):
        run_activation_preflight(manifest, seeds=(111, 112), scenario_id=PREFLIGHT_SCENARIO)


def test_preflight_rejects_non_frozen_scenario() -> None:
    """The preflight must exercise a frozen scenario."""
    manifest = _manifest()
    with pytest.raises(campaign.PreflightActivationError, match="frozen scenarios"):
        run_activation_preflight(manifest, seeds=DISJOINT_SEEDS, scenario_id="not_a_scenario")


def test_preflight_runs_natively_and_satisfies_activation_gate() -> None:
    """The disjoint-seed preflight binds all three tiers and activates non-nominal tiers.

    This is the real end-to-end activation check: it builds the real bicycle-drive
    env through the canonical runner's variant applier and action converter, drives
    the robot at the tier cap, and confirms the speed-cap intervention measurably
    activates for the 3.0 and 4.0 m/s tiers. It uses only disjoint seeds.
    """
    manifest = _manifest()
    preflight = run_activation_preflight(
        manifest,
        seeds=DISJOINT_SEEDS,
        scenario_id=PREFLIGHT_SCENARIO,
        steps=80,
    )
    assert preflight["not_evidence_banner"] == NOT_EVIDENCE_BANNER
    assert preflight["preflight_passed"] is True
    assert preflight["execution_status"]["native"] is True
    assert preflight["execution_status"]["fallback"] is False
    assert preflight["execution_status"]["degraded"] is False
    assert preflight["activation_probe"]["seeds_disjoint_from_registered_111_140"] is True
    assert set(preflight["activation_probe"]["seeds"]).isdisjoint(set(DECLARED_SEEDS))
    # Planned versus resolved cap must match for every tier.
    for entry in preflight["planned_vs_resolved"]:
        assert entry["cap_matches"] is True
    gates = preflight["activation_gate_summary"]["per_tier"]
    assert gates["cap_3_0"]["activated"] is True
    assert gates["cap_4_0"]["activated"] is True
    # Non-nominal tiers must measurably exceed the 2.0 m/s boundary.
    for tier_id in NON_NOMINAL_TIERS:
        peak = next(
            t["realized_speed_peak_m_s"]
            for t in preflight["tier_results"]
            if t["tier_id"] == tier_id
        )
        assert peak > MIN_ACTIVATION_PEAK_SPEED


def test_preflight_artifact_carries_provenance_and_command_manifest(tmp_path: Path) -> None:
    """The preflight artifact records git SHA, command manifest, and diagnostics."""
    manifest = _manifest()
    preflight = run_activation_preflight(
        manifest,
        seeds=DISJOINT_SEEDS,
        scenario_id=PREFLIGHT_SCENARIO,
        steps=60,
        git_provenance={
            "git_head": "deadbeef",
            "git_worktree_dirty": False,
            "git_status_short": [],
        },
    )
    artifact_dir = tmp_path / "preflight"
    campaign.write_preflight_artifact(preflight, artifact_dir)
    decoded = json.loads(
        (artifact_dir / "issue_5578_activation_preflight.json").read_text(encoding="utf-8")
    )
    assert decoded["git_provenance"]["git_head"] == "deadbeef"
    assert "command_environment_manifest" in decoded
    assert decoded["command_environment_manifest"]["env_factory"].endswith("make_robot_env")
    assert len(decoded["tier_results"]) == 3
    assert decoded["activation_rule"]["min_fraction_above_2_0_mps"] == 0.05


def test_preflight_seeds_are_outside_registered_block() -> None:
    """The default preflight seeds are disjoint from the registered 111-140 block."""
    assert set(PREFLIGHT_SEEDS).isdisjoint(set(DECLARED_SEEDS))
    assert all(seed < 111 or seed > 140 for seed in PREFLIGHT_SEEDS)


@pytest.mark.parametrize(
    "tier_id,cap,accel,decel",
    [(NOMINAL_TIER_ID, 2.0, 1.0, 2.0), ("cap_3_0", 3.0, 1.5, 3.0), ("cap_4_0", 4.0, 2.0, 4.0)],
)
def test_frozen_tier_speed_cap_binds_to_drive_model(
    tier_id: str, cap: float, accel: float, decel: float
) -> None:
    """The frozen #5578 tier cap flows to the drive model through the canonical applier."""
    from types import SimpleNamespace

    from robot_sf.robot.bicycle_drive import BicycleDriveSettings
    from scripts.benchmark.run_fidelity_sensitivity_campaign import (
        VariantSpec,
        _robot_speed_cap,
        apply_variant,
    )

    config = SimpleNamespace(robot_config=BicycleDriveSettings())
    variant = VariantSpec(
        axis="robot_speed_band",
        key=f"issue_5578_{tier_id}",
        source_key=f"issue_5578_{tier_id}",
        baseline=tier_id == NOMINAL_TIER_ID,
        patch={
            "robot_config": {
                "type": "bicycle_drive",
                "max_velocity": cap,
                "max_accel": accel,
                "max_decel": decel,
            }
        },
        observation_noise={},
        runtime_binding="robot_config.drive_speed_cap",
    )
    apply_variant(config, variant, seed=211)
    assert _robot_speed_cap(config.robot_config) == pytest.approx(cap)
    assert config.robot_config.max_velocity == pytest.approx(cap)
    assert config.robot_config.max_accel == pytest.approx(accel)
    assert config.robot_config.max_decel == pytest.approx(decel)


def _smoke_rows(seed: int = 111) -> list[dict[str, Any]]:
    """Build a minimal synthesizer-conformant row set for adapter smoke checks."""
    rows: list[dict[str, Any]] = []
    for tier_id, cap in (
        (NOMINAL_TIER_ID, 2.0),
        ("cap_3_0", 3.0),
        ("cap_4_0", 4.0),
    ):
        rows.append(
            {
                "scenario_id": PREFLIGHT_SCENARIO,
                "speed_tier_id": tier_id,
                "speed_cap_m_s": cap,
                "planner_id": "orca",
                "seed": seed,
                "horizon_steps": 600,
                "dt_seconds": 0.1,
                "execution_mode": "native",
                "success_rate": 0.8,
                "collision_rate": 0.1,
                "near_miss_rate": 0.2,
                "ped_collision_rate": 0.0,
                "obstacle_collision_rate": 0.0,
                "agent_collision_rate": 0.0,
                "unclassified_collision_rate": 0.0,
                "commanded_speed_mean_m_s": cap * 0.9,
                "realized_speed_mean_m_s": cap * 0.85,
                "realized_speed_peak_m_s": cap,
                "fraction_above_2_0_mps": 0.5 if cap > 2.0 else 0.0,
                "cap_saturation_fraction": 0.3,
                "resolved_actuation_envelope": {
                    "drive_model": "bicycle_drive",
                    "max_forward_accel_m_s2": cap * 0.5,
                    "max_braking_decel_m_s2": cap,
                    "peak_forward_speed_m_s": cap,
                    "stopping_distance_envelope_m": cap * 0.5,
                },
                "time_to_goal_norm": 0.5,
                "total_exposure_seconds": 30.0,
                "travel_distance_m": 60.0,
                "mean_clearance_m": 1.2,
                "min_clearance_m": 0.4,
            }
        )
    return rows
