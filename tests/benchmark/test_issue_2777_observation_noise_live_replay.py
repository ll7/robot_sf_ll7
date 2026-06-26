"""Tests for the issue #2777 live observation-perturbation replay wrapper."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml

from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_observation_noise_live_replay_issue_2777.py"
_LOADED_MOD = None


def _load_script():
    """Load the issue script as a module."""
    global _LOADED_MOD
    if _LOADED_MOD is not None:
        return _LOADED_MOD
    spec = importlib.util.spec_from_file_location(
        "run_observation_noise_live_replay_issue_2777", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_observation_noise_live_replay_issue_2777"] = mod
    spec.loader.exec_module(mod)
    _LOADED_MOD = mod
    return mod


def test_condition_set_matches_issue_2755_contract() -> None:
    """The wrapper should preserve the default perturbation family names."""
    mod = _load_script()

    assert tuple(condition.name for condition in mod.CONDITIONS) == mod.REQUIRED_CONDITIONS
    assert mod.CONDITION_SETS[mod.DEFAULT_CONDITION_SET] == mod.CONDITIONS
    delay = next(condition for condition in mod.CONDITIONS if condition.name == "delay_only")
    assert "--observation-delay-steps" in delay.flags
    assert "2" in delay.flags
    false_positive = next(
        condition for condition in mod.CONDITIONS if condition.name == "false_positive_only"
    )
    assert false_positive.flags == (
        "--false-positive-actor-count",
        "1",
        "--false-positive-offset-x-m",
        "1.0",
        "--false-positive-offset-y-m",
        "0.0",
        "--observation-perturbation-seed",
        "2755",
    )


def test_issue_3328_behavior_probe_condition_set_is_opt_in() -> None:
    """The #3328 probe should not mutate the default condition run."""
    mod = _load_script()

    probe_conditions = mod.CONDITION_SETS[mod.ISSUE_3328_CONDITION_SET]

    assert tuple(condition.name for condition in probe_conditions) == (
        "noop",
        "medium_noise",
        "delay_only",
        "high_noise_3328",
    )
    high_noise = next(
        condition for condition in probe_conditions if condition.name == "high_noise_3328"
    )
    assert high_noise.flags == (
        "--observation-noise-std-m",
        "1.0",
        "--observation-noise-bound-m",
        "2.0",
        "--observation-perturbation-seed",
        "3328",
    )
    assert tuple(condition.name for condition in mod.CONDITIONS) == mod.REQUIRED_CONDITIONS


def test_issue_3330_seed_amplitude_grid_condition_set_is_opt_in_and_ordered() -> None:
    """The #3330 grid should preserve defaults while varying seed and amplitude."""
    mod = _load_script()

    grid_conditions = mod.CONDITION_SETS[mod.ISSUE_3330_CONDITION_SET]

    assert tuple(condition.name for condition in grid_conditions) == (
        "noop",
        "delay_only",
        "medium_noise_2755",
        "medium_noise_3328",
        "medium_noise_3330",
        "high_noise_2755",
        "high_noise_3328",
        "high_noise_3330",
    )
    expected_flags = {
        "medium_noise_2755": ("0.30", "0.60", "2755"),
        "medium_noise_3328": ("0.30", "0.60", "3328"),
        "medium_noise_3330": ("0.30", "0.60", "3330"),
        "high_noise_2755": ("1.00", "2.00", "2755"),
        "high_noise_3328": ("1.00", "2.00", "3328"),
        "high_noise_3330": ("1.00", "2.00", "3330"),
    }
    for condition in grid_conditions:
        if condition.name in expected_flags:
            std, bound, seed = expected_flags[condition.name]
            assert condition.flags == (
                "--observation-noise-std-m",
                std,
                "--observation-noise-bound-m",
                bound,
                "--observation-perturbation-seed",
                seed,
            )
    assert tuple(condition.name for condition in mod.CONDITIONS) == mod.REQUIRED_CONDITIONS
    assert (
        tuple(condition.name for condition in mod.CONDITION_SETS[mod.ISSUE_3328_CONDITION_SET])
        == mod.ISSUE_3328_REQUIRED_CONDITIONS
    )


def test_default_strict_mode_fails_closed_without_occluded_fixture(tmp_path: Path) -> None:
    """A non-occluded matrix should not be promoted as #2777 live replay evidence."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "select_scenarios: [issue_3233_near_field_observation_noise]\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    args = mod.parse_args(["--scenario-matrix", str(matrix), "--output-dir", str(output_dir)])

    report = mod.run_live_batch(args)

    assert report["status"] == "fail_closed"
    assert report["classification"]["label"] == "blocked"
    assert report["fixture_contract"]["satisfied"] is False
    assert len(report["conditions"]) == len(mod.REQUIRED_CONDITIONS)
    assert {condition["status"] for condition in report["conditions"]} == {"blocked"}


def test_issue_3320_matrix_satisfies_occluded_emergence_contract() -> None:
    """The tracked issue #3320 matrix preserves the #2756 fixture boundary."""
    mod = _load_script()
    matrix = mod.REPO_ROOT / "configs/scenarios/sets/issue_3320_occluded_emergence_live_replay.yaml"

    contract = mod._fixture_contract(matrix)

    assert contract["satisfied"] is True
    assert contract["blocker"] is None
    assert contract["matched_scenario"]["name"] == "issue_2756_occluded_emergence"
    assert contract["matched_scenario"]["seeds"] == [111]
    assert contract["matched_scenario"]["first_visible_step"] == 5
    assert contract["matched_scenario"]["delay_steps"] == 2
    assert contract["matched_scenario"]["delay_only_expected_first_observed_step"] == 7


def test_issue_3323_matrix_adds_near_field_route_and_preserves_fixture_contract() -> None:
    """The issue #3323 matrix should keep the #2756 boundary with a near-field robot route."""
    mod = _load_script()
    matrix = (
        mod.REPO_ROOT
        / "configs/scenarios/sets/issue_3323_occluded_emergence_near_field_live_replay.yaml"
    )
    route_override = (
        mod.REPO_ROOT / "configs/scenarios/route_overrides/issue_3323/"
        "occluded_emergence_near_field_h1.yaml"
    )

    contract = mod._fixture_contract(matrix)

    assert contract["satisfied"] is True
    assert contract["matched_scenario"]["name"] == "issue_2756_occluded_emergence"
    assert contract["matched_scenario"]["seeds"] == [111]
    assert contract["matched_scenario"]["first_visible_step"] == 5
    assert contract["matched_scenario"]["delay_only_expected_first_observed_step"] == 7
    assert route_override.exists()

    route_payload = yaml.safe_load(route_override.read_text(encoding="utf-8"))["route_payload"]
    assert route_payload["robot_routes"] == [
        {"spawn_id": 1, "goal_id": 1, "waypoints": [[27.0, 9.0], [27.0, 4.5]]}
    ]

    scenarios = load_scenarios(matrix)
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["name"] == "issue_2756_occluded_emergence"
    route_overrides_path = Path(str(scenario["route_overrides_file"]))
    if not route_overrides_path.is_absolute():
        route_overrides_path = (matrix.parent / route_overrides_path).resolve()
    assert route_overrides_path == route_override

    config = build_robot_config_from_scenario(scenario, scenario_path=matrix.resolve())
    _map_name, map_def = next(iter(config.map_pool.map_defs.items()))
    assert map_def.robot_routes[0].waypoints == [(27.0, 9.0), (27.0, 4.5)]
    assert map_def.robot_routes[0].spawn_zone[0] == (27.0, 9.0)
    assert map_def.robot_routes[0].goal_zone[0] == (27.0, 4.5)


def test_mismatched_occluded_emergence_contract_fails_closed(tmp_path: Path) -> None:
    """A named #2756 scenario must still preserve the exact boundary metadata."""
    mod = _load_script()
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "scenarios:\n"
        "- name: issue_2756_occluded_emergence\n"
        "  metadata:\n"
        "    source_issue: 2756\n"
        "    family: occluded_emergence\n"
        "    label: deterministic_occluded_emergence\n"
        "    fixture_contract:\n"
        "      first_visible_step: 4\n"
        "      delay_steps: 2\n"
        "      delay_only_expected_first_observed_step: 6\n"
        "  seeds: [111]\n",
        encoding="utf-8",
    )
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "includes:\n"
        "  - scenario.yaml\n"
        "select_scenarios:\n"
        "  - issue_2756_occluded_emergence\n",
        encoding="utf-8",
    )

    contract = mod._fixture_contract(matrix)

    assert contract["satisfied"] is False
    assert "first_visible_step" in contract["blocker"]
    assert "delay_only_expected_first_observed_step" in contract["blocker"]


def test_dry_run_plans_all_live_diagnostics_commands(tmp_path: Path) -> None:
    """Proxy dry-run mode should expose the exact subprocess plan without execution."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "select_scenarios: [issue_3233_near_field_observation_noise]\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    args = mod.parse_args(
        [
            "--scenario-matrix",
            str(matrix),
            "--output-dir",
            str(output_dir),
            "--allow-non-occluded-live-fixture",
            "--dry-run",
        ]
    )

    report = mod.run_live_batch(args)

    assert report["status"] == "diagnostic_only"
    assert [condition["name"] for condition in report["conditions"]] == list(
        mod.REQUIRED_CONDITIONS
    )
    commands = {condition["name"]: condition["command"] for condition in report["conditions"]}
    assert "--observation-delay-steps" in commands["delay_only"]
    assert "--occlusion-distance-m" in commands["occlusion_only"]
    assert "--missed-detection-probability" in commands["missed_detection_only"]
    assert (output_dir / "generated_policy_search_funnel.yaml").exists()


def test_issue_3328_behavior_probe_dry_run_plans_only_probe_conditions(tmp_path: Path) -> None:
    """The opt-in #3328 condition set should plan only the probe conditions."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "select_scenarios: [issue_3233_near_field_observation_noise]\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    args = mod.parse_args(
        [
            "--condition-set",
            mod.ISSUE_3328_CONDITION_SET,
            "--scenario-matrix",
            str(matrix),
            "--output-dir",
            str(output_dir),
            "--allow-non-occluded-live-fixture",
            "--dry-run",
        ]
    )

    report = mod.run_live_batch(args)

    assert report["run_config"]["condition_set"] == mod.ISSUE_3328_CONDITION_SET
    assert [condition["name"] for condition in report["conditions"]] == list(
        mod.ISSUE_3328_REQUIRED_CONDITIONS
    )
    commands = {condition["name"]: condition["command"] for condition in report["conditions"]}
    assert "--observation-noise-std-m" in commands["high_noise_3328"]
    assert "1.0" in commands["high_noise_3328"]
    assert "--observation-noise-bound-m" in commands["high_noise_3328"]
    assert "2.0" in commands["high_noise_3328"]
    assert "--observation-perturbation-seed" in commands["high_noise_3328"]
    assert "3328" in commands["high_noise_3328"]


def test_issue_3330_seed_amplitude_grid_dry_run_plans_grid_conditions(
    tmp_path: Path,
) -> None:
    """The opt-in #3330 grid should plan only the seed/amplitude grid."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "select_scenarios: [issue_3233_near_field_observation_noise]\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    args = mod.parse_args(
        [
            "--condition-set",
            mod.ISSUE_3330_CONDITION_SET,
            "--scenario-matrix",
            str(matrix),
            "--output-dir",
            str(output_dir),
            "--allow-non-occluded-live-fixture",
            "--dry-run",
        ]
    )

    report = mod.run_live_batch(args)

    assert report["run_config"]["condition_set"] == mod.ISSUE_3330_CONDITION_SET
    assert [condition["name"] for condition in report["conditions"]] == list(
        mod.ISSUE_3330_REQUIRED_CONDITIONS
    )
    assert "grid_interpretation" not in report
    commands = {condition["name"]: condition["command"] for condition in report["conditions"]}
    assert "2755" in commands["medium_noise_2755"]
    assert "3328" in commands["medium_noise_3328"]
    assert "3330" in commands["medium_noise_3330"]
    assert "1.00" in commands["high_noise_3330"]
    assert "2.00" in commands["high_noise_3330"]


def test_scalar_includes_are_ignored_for_fixture_detection(tmp_path: Path) -> None:
    """Malformed scalar includes should not be iterated character-by-character."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "includes: ../single/issue_2756_occluded_emergence.yaml\n",
        encoding="utf-8",
    )

    contract = mod._fixture_contract(matrix)

    assert contract["satisfied"] is False
    assert "blocker" in contract


def test_invalid_yaml_matrix_fails_closed_even_with_fixture_token(tmp_path: Path) -> None:
    """Invalid matrix syntax should not satisfy fixture detection by raw text."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "select_scenarios: [issue_2756_occluded_emergence\n",
        encoding="utf-8",
    )

    contract = mod._fixture_contract(matrix)

    assert contract["satisfied"] is False
    assert "not valid YAML" in contract["blocker"]


def test_live_condition_timeout_becomes_fail_closed_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Subprocess timeouts should produce a compact blocker instead of crashing."""
    mod = _load_script()

    def _raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["diagnostics"], timeout=0.01, stderr="hung")

    monkeypatch.setattr(mod.subprocess, "run", _raise_timeout)
    args = mod.parse_args(
        [
            "--scenario-matrix",
            str(tmp_path / "matrix.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--allow-non-occluded-live-fixture",
        ]
    )
    funnel = mod._write_generated_funnel(
        output_dir=args.output_dir,
        scenario_matrix=args.scenario_matrix,
        stage=args.stage,
        horizon=args.horizon,
    )

    conditions, blockers = mod._execute_conditions(
        args=args,
        output_dir=args.output_dir,
        funnel_config=funnel,
    )

    assert len(conditions) == len(mod.REQUIRED_CONDITIONS)
    assert len(blockers) == len(mod.REQUIRED_CONDITIONS)
    assert all(condition["status"] == "blocked" for condition in conditions)
    assert "timed out" in blockers[0]


def _write_trace(
    path: Path,
    *,
    command: list[float],
    observed_count: int,
    closest: float,
    seed: int = 111,
    false_positive_count: int = 0,
) -> None:
    payload = {
        "candidate": "risk_surface_dwa_v0",
        "stage": "issue_2777_live_replay",
        "scenario_id": "issue_2756_occluded_emergence",
        "seed": seed,
        "algo": "risk_surface_dwa",
        "progress_summary": {
            "net_goal_progress": 1.0,
            "best_goal_progress": 1.2,
            "closest_robot_ped_distance": closest,
            "closest_robot_ped_step": 3,
            "collision_flag_counts": {"pedestrian": 0, "obstacle": 0, "robot": 0},
            "progress_step_count": 1,
            "regression_step_count": 0,
            "stagnant_step_count": 0,
            "longest_stagnant_run": 0,
        },
        "steps": [
            {
                "policy_command": command,
                "observation_perturbation": {
                    "noise_profile": "bounded_gaussian",
                    "missed_actor_count": 0,
                    "occluded_actor_count": 0,
                    "false_positive_actor_count": false_positive_count,
                    "observed_actor_count": observed_count,
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_observed_count_trace(path: Path, counts: list[int], closest: float = 1.5) -> None:
    """Write a tiny diagnostics trace with configurable observed actor counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_id": "issue_2756_occluded_emergence",
        "seed": 111,
        "progress_summary": {
            "closest_robot_ped_distance": closest,
        },
        "steps": [
            {
                "step": step,
                "observation_perturbation": {
                    "observed_actor_count": count,
                },
            }
            for step, count in enumerate(counts)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_grid_trace(
    path: Path,
    *,
    command: list[float],
    first_observed_step: int,
    closest: float = 1.5,
) -> None:
    """Write a compact multi-step trace for #3330 live-report tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidate": "risk_surface_dwa_v0",
        "stage": "issue_2777_live_replay",
        "scenario_id": "issue_2756_occluded_emergence",
        "seed": 111,
        "algo": "risk_surface_dwa",
        "progress_summary": {
            "net_goal_progress": 1.0,
            "best_goal_progress": 1.2,
            "closest_robot_ped_distance": closest,
            "closest_robot_ped_step": 3,
            "collision_flag_counts": {"pedestrian": 0, "obstacle": 0, "robot": 0},
            "progress_step_count": 1,
            "regression_step_count": 0,
            "stagnant_step_count": 0,
            "longest_stagnant_run": 0,
        },
        "steps": [
            {
                "step": step,
                "policy_command": command,
                "observation_perturbation": {
                    "noise_profile": "bounded_gaussian",
                    "missed_actor_count": 0,
                    "occluded_actor_count": 0,
                    "observed_actor_count": 1 if step >= first_observed_step else 0,
                },
            }
            for step in range(8)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_fixture_boundary_verifier_requires_noop_and_delay_first_observed_steps(
    tmp_path: Path,
) -> None:
    """The live wrapper should fail closed when trace timing drifts from #2756."""
    mod = _load_script()
    output_dir = tmp_path / "out"
    contract = {
        "satisfied": True,
        "first_visible_step": 5,
        "delay_only_expected_first_observed_step": 7,
    }
    _write_observed_count_trace(output_dir / "traces/noop/trace.json", [0, 0, 0, 0, 0, 1])
    _write_observed_count_trace(
        output_dir / "traces/delay_only/trace.json",
        [0, 0, 0, 0, 0, 0, 0, 1],
    )

    assert (
        mod._verify_fixture_observation_boundary(
            output_dir=output_dir,
            fixture_contract=contract,
        )
        == []
    )

    _write_observed_count_trace(
        output_dir / "traces/delay_only/trace.json",
        [0, 0, 0, 0, 0, 0, 1],
    )
    blockers = mod._verify_fixture_observation_boundary(
        output_dir=output_dir,
        fixture_contract=contract,
    )

    assert blockers
    assert "Delay-only" in blockers[0]


def test_issue_3330_live_report_includes_grid_interpretation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A completed #3330 live grid should include compact diagnostic interpretation."""
    mod = _load_script()

    def _fake_execute_conditions(
        *,
        args,
        output_dir: Path,
        funnel_config: Path,
    ) -> tuple[list[dict[str, object]], list[str]]:
        del args, funnel_config
        conditions: list[dict[str, object]] = []
        for name in mod.ISSUE_3330_REQUIRED_CONDITIONS:
            command = [0.0, 0.0] if name == "medium_noise_3328" else [1.0, 0.0]
            first_observed_step = 7 if name == "delay_only" else 5
            trace_path = output_dir / "traces" / name / "trace.json"
            _write_grid_trace(
                trace_path,
                command=command,
                first_observed_step=first_observed_step,
            )
            conditions.append(
                {
                    "name": name,
                    "description": name,
                    "status": "live_replay",
                    "trace": str(trace_path),
                }
            )
        return conditions, []

    monkeypatch.setattr(mod, "_execute_conditions", _fake_execute_conditions)
    args = mod.parse_args(
        [
            "--condition-set",
            mod.ISSUE_3330_CONDITION_SET,
            "--scenario-matrix",
            str(
                mod.REPO_ROOT
                / "configs/scenarios/sets/issue_3320_occluded_emergence_live_replay.yaml"
            ),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    report = mod.run_live_batch(args)
    markdown = mod._markdown(report)

    assert report["status"] == "live_replay"
    assert report["grid_interpretation"]["label"] == "medium_amplitude_sensitive"
    assert report["grid_interpretation"]["evidence_status"] == "diagnostic-only"
    assert "## Grid Interpretation" in markdown
    assert "medium-amplitude-sensitive" in markdown


def test_issue_3328_behavior_probe_guardrails_require_near_field_fixture(
    tmp_path: Path,
) -> None:
    """The opt-in probe should fail closed if the no-op trace is not near-field."""
    mod = _load_script()
    output_dir = tmp_path / "out"
    contract = {
        "satisfied": True,
        "matched_scenario": {
            "name": "issue_2756_occluded_emergence",
            "seeds": [111],
        },
    }
    _write_observed_count_trace(
        output_dir / "traces/noop/trace.json",
        [0, 0, 0, 0, 0, 1],
        closest=1.9,
    )
    _write_observed_count_trace(
        output_dir / "traces/delay_only/trace.json",
        [0, 0, 0, 0, 0, 0, 0, 1],
        closest=1.9,
    )

    assert (
        mod._verify_issue_3328_behavior_probe_guardrails(
            output_dir=output_dir,
            fixture_contract=contract,
        )
        == []
    )

    _write_observed_count_trace(
        output_dir / "traces/noop/trace.json",
        [0, 0, 0, 0, 0, 1],
        closest=2.1,
    )
    blockers = mod._verify_issue_3328_behavior_probe_guardrails(
        output_dir=output_dir,
        fixture_contract=contract,
    )

    assert blockers
    assert "closest_robot_ped_distance <= 2.0" in blockers[-1]

    _write_observed_count_trace(
        output_dir / "traces/noop/trace.json",
        [0, 0, 0, 0, 0, 1],
        closest=float("nan"),
    )
    blockers = mod._verify_issue_3328_behavior_probe_guardrails(
        output_dir=output_dir,
        fixture_contract=contract,
    )

    assert blockers
    assert "finite no-op closest_robot_ped_distance" in blockers[-1]


def test_issue_3330_seed_amplitude_grid_reuses_near_field_fixture_guardrails(
    tmp_path: Path,
) -> None:
    """The #3330 grid should fail closed on the same near-field fixture contract."""
    mod = _load_script()
    output_dir = tmp_path / "out"
    contract = {
        "satisfied": True,
        "matched_scenario": {
            "name": "issue_2756_occluded_emergence",
            "seeds": [111],
        },
    }
    _write_observed_count_trace(
        output_dir / "traces/noop/trace.json",
        [0, 0, 0, 0, 0, 1],
        closest=1.9,
    )
    _write_observed_count_trace(
        output_dir / "traces/delay_only/trace.json",
        [0, 0, 0, 0, 0, 0, 0, 1],
        closest=1.9,
    )

    assert (
        mod._verify_issue_3330_seed_amplitude_grid_guardrails(
            output_dir=output_dir,
            fixture_contract=contract,
        )
        == []
    )

    _write_observed_count_trace(
        output_dir / "traces/noop/trace.json",
        [0, 0, 0, 0, 0, 1],
        closest=2.1,
    )
    blockers = mod._verify_issue_3330_seed_amplitude_grid_guardrails(
        output_dir=output_dir,
        fixture_contract=contract,
    )

    assert blockers
    assert any("Issue #3330 seed/amplitude grid" in blocker for blocker in blockers)
    assert any("closest_robot_ped_distance <= 2.0" in blocker for blocker in blockers)


def _grid_condition_row(
    name: str,
    label: str,
    status: str = "live_replay",
) -> dict[str, object]:
    """Return a compact condition row for grid interpretation tests."""
    return {
        "name": name,
        "status": status,
        "classification": {"label": label},
    }


def _grid_rows(
    *,
    medium_sensitive: set[str] | None = None,
    high_sensitive: set[str] | None = None,
) -> list[dict[str, object]]:
    """Return complete #3330 rows with selected behavior-sensitive conditions."""
    medium_sensitive = medium_sensitive or set()
    high_sensitive = high_sensitive or set()
    rows = [
        _grid_condition_row("noop", "baseline"),
        _grid_condition_row("delay_only", "policy_insensitive"),
    ]
    for seed in ("2755", "3328", "3330"):
        name = f"medium_noise_{seed}"
        label = (
            "behavior_sensitive_diagnostic_only"
            if name in medium_sensitive
            else "policy_insensitive"
        )
        rows.append(_grid_condition_row(name, label))
    for seed in ("2755", "3328", "3330"):
        name = f"high_noise_{seed}"
        label = (
            "behavior_sensitive_diagnostic_only" if name in high_sensitive else "policy_insensitive"
        )
        rows.append(_grid_condition_row(name, label))
    return rows


def test_issue_3330_grid_interpretation_labels_seed_amplitude_patterns() -> None:
    """The grid-level summary should stay diagnostic-only and compact."""
    mod = _load_script()

    cases = [
        (
            {"medium_noise_3328"},
            set(),
            "medium_amplitude_sensitive",
            "medium-amplitude-sensitive",
        ),
        (
            set(),
            {"high_noise_2755", "high_noise_3328", "high_noise_3330"},
            "high_noise_persistent",
            "high-noise persistent",
        ),
        (
            set(),
            {"high_noise_2755", "high_noise_3330"},
            "mixed_seed_specific",
            "mixed/seed-specific",
        ),
        (set(), set(), "not_reproduced", "not reproduced"),
    ]
    for medium_sensitive, high_sensitive, label, expected_phrase in cases:
        interpretation = mod._issue_3330_grid_interpretation(
            conditions=_grid_rows(
                medium_sensitive=medium_sensitive,
                high_sensitive=high_sensitive,
            ),
            blockers=[],
        )

        assert interpretation["label"] == label
        assert interpretation["evidence_status"] == "diagnostic-only"
        assert expected_phrase in interpretation["summary"]


def test_issue_3330_grid_interpretation_fails_closed_when_unavailable() -> None:
    """The grid-level summary should not interpret incomplete live evidence."""
    mod = _load_script()
    rows = _grid_rows()
    rows[-1]["status"] = "blocked"

    interpretation = mod._issue_3330_grid_interpretation(
        conditions=rows,
        blockers=["high_noise_3330 live replay failed"],
    )

    assert interpretation["label"] == "unavailable_fail_closed"
    assert "unavailable/fail-closed" in interpretation["summary"]
    assert interpretation["sensitive_conditions"] == []


def test_trace_comparison_names_scenario_seed_planner_and_policy_insensitive(
    tmp_path: Path,
) -> None:
    """Comparison payload should carry the benchmark-facing provenance fields."""
    mod = _load_script()
    noop = tmp_path / "noop.json"
    condition = tmp_path / "condition.json"
    _write_trace(noop, command=[1.0, 0.0], observed_count=1, closest=1.5)
    _write_trace(condition, command=[1.0, 0.0], observed_count=0, closest=1.5)

    comparison = mod._compare_condition(
        noop_trace_path=noop,
        condition_trace_path=condition,
        fixture_contract_satisfied=True,
    )

    assert comparison["scenario"]["same"] is True
    assert comparison["seed"]["same"] is True
    assert comparison["planner_mode"]["candidate"] == "risk_surface_dwa_v0"
    assert comparison["classification"]["label"] == "policy_insensitive"


def test_observation_totals_report_false_positive_actor_injection(tmp_path: Path) -> None:
    """Comparison metadata should expose observed-only actor injection counts."""
    mod = _load_script()
    trace = tmp_path / "false_positive.json"
    _write_trace(
        trace,
        command=[1.0, 0.0],
        observed_count=2,
        closest=1.5,
        false_positive_count=1,
    )
    payload = json.loads(trace.read_text(encoding="utf-8"))

    totals = mod._observation_totals(payload)

    assert totals["false_positive_actor_observations_total"] == 1
    assert totals["max_observed_actor_count"] == 2


def test_trace_comparison_reports_behavior_change_dimensions(tmp_path: Path) -> None:
    """Comparison payload should name command, risk, min-distance, collision, and stop proxies."""
    mod = _load_script()
    noop = tmp_path / "noop.json"
    condition = tmp_path / "condition.json"
    _write_trace(noop, command=[0.0, 0.0], observed_count=1, closest=1.5)
    _write_trace(condition, command=[1.0, 0.0], observed_count=0, closest=0.8)
    noop_payload = json.loads(noop.read_text(encoding="utf-8"))
    condition_payload = json.loads(condition.read_text(encoding="utf-8"))
    noop_payload["progress_summary"]["collision_flag_counts"]["pedestrian"] = 0
    condition_payload["progress_summary"]["collision_flag_counts"]["pedestrian"] = 1
    noop_payload["steps"][0]["step"] = 3
    condition_payload["steps"][0]["step"] = 3
    noop_payload["steps"][0]["meta"] = {"near_misses": 0}
    condition_payload["steps"][0]["meta"] = {"near_misses": 1}
    noop.write_text(json.dumps(noop_payload), encoding="utf-8")
    condition.write_text(json.dumps(condition_payload), encoding="utf-8")

    comparison = mod._compare_condition(
        noop_trace_path=noop,
        condition_trace_path=condition,
        fixture_contract_satisfied=True,
    )

    behavior = comparison["behavior_change_summary"]
    assert behavior["command_sequence_changed"] is True
    assert behavior["changed_command_steps"] == [3]
    assert comparison["command_summary"]["changed_steps"] == [3]
    assert behavior["progress_or_risk_changed"] is True
    assert "collision_flag_counts" in behavior["progress_delta_changed_fields"]
    assert behavior["min_distance_changed"] is True
    assert behavior["closest_robot_ped"]["distance"]["condition"] == 0.8
    assert behavior["collision_or_near_miss_changed"] is True
    assert behavior["collision_summary"]["changed"] is True
    assert behavior["near_miss_summary"]["status"] == "available"
    assert behavior["near_miss_summary"]["changed"] is True
    assert behavior["stop_yield_timing_proxy"]["status"] == "available"
    assert behavior["stop_yield_timing_proxy"]["direct_event_available"] is False
    assert "policy_command linear speed" in behavior["stop_yield_timing_proxy"]["definition"]
    assert behavior["stop_yield_timing_proxy"]["changed"] is True
    assert behavior["stop_yield_timing_proxy"]["noop_first_stop_step"] == 3
    assert behavior["stop_yield_timing_proxy"]["condition_first_stop_step"] is None
    assert behavior["stop_yield_timing"]["direct_event_available"] is False
    assert "not available" in behavior["stop_yield_timing"]["limitation"]
    assert behavior["stop_yield_timing"]["command_stop_proxy"]["changed"] is True
    assert behavior["stop_yield_timing"]["command_stop_proxy"]["noop_first_stop_step"] == 3
    assert behavior["stop_yield_timing"]["command_stop_proxy"]["condition_first_stop_step"] is None
    assert comparison["classification"]["label"] == "behavior_sensitive_diagnostic_only"


def test_progress_delta_treats_paired_nan_values_as_unchanged() -> None:
    """Paired NaN summary fields should not create false behavior sensitivity."""
    mod = _load_script()
    noop = {"progress_summary": {"net_goal_progress": float("nan")}}
    condition = {"progress_summary": {"net_goal_progress": float("nan")}}

    delta = mod._progress_delta(noop, condition)

    assert delta["net_goal_progress"]["changed"] is False


def test_near_miss_summary_ignores_non_finite_counts() -> None:
    """Non-finite near-miss metadata should not poison the summed comparison."""
    mod = _load_script()
    noop = {
        "steps": [
            {"meta": {"near_misses": 1}},
            {"meta": {"near_misses": float("nan")}},
        ]
    }
    condition = {
        "steps": [
            {"meta": {"near_misses": 1}},
            {"meta": {"near_misses": float("inf")}},
        ]
    }

    summary = mod._near_miss_summary(noop, condition)

    assert summary["status"] == "available"
    assert summary["noop"] == 1
    assert summary["condition"] == 1
    assert summary["changed"] is False


def test_observation_totals_ignore_missing_noise_profile(tmp_path: Path) -> None:
    """A null noise profile should not become a literal profile label."""
    mod = _load_script()
    trace = tmp_path / "trace.json"
    _write_trace(trace, command=[1.0, 0.0], observed_count=1, closest=1.5)
    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["steps"][0]["observation_perturbation"]["noise_profile"] = None

    totals = mod._observation_totals(payload)

    assert totals["noise_profiles"] == []


def test_markdown_and_json_outputs_are_written(tmp_path: Path) -> None:
    """The wrapper writes compact durable artifacts."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("schema_version: robot_sf.scenario_matrix.v1\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = mod.main(["--scenario-matrix", str(matrix), "--output-dir", str(output_dir)])

    assert exit_code == 2
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == mod.SCHEMA_VERSION
    assert "Issue #2777" in (output_dir / "README.md").read_text(encoding="utf-8")


def test_relative_output_dir_is_repo_relative_from_other_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Repository-relative artifact paths should not depend on the launch cwd."""
    mod = _load_script()
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    matrix = fake_repo / "matrix.yaml"
    matrix.write_text("schema_version: robot_sf.scenario_matrix.v1\n", encoding="utf-8")
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo)
    monkeypatch.chdir(outside_cwd)

    exit_code = mod.main(["--scenario-matrix", "matrix.yaml", "--output-dir", "relative-out"])

    assert exit_code == 2
    assert (fake_repo / "relative-out" / "summary.json").exists()
    assert not (outside_cwd / "relative-out").exists()
