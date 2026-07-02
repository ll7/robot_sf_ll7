"""Tests issue #3300 false-positive actor-injection replay report."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from robot_sf.benchmark.false_positive_replay_report import (
    CLASS_BLOCKED_UNAVAILABLE,
    CLASS_OBSERVED,
    CLASS_SCENARIO_TOO_WEAK,
    CLASS_TRACE_ONLY_DIAGNOSTIC,
    build_false_positive_replay_report,
    classify_false_positive_replay,
)
from robot_sf.benchmark.observation_noise import (
    apply_observation_noise,
    make_observation_noise_rng,
    normalize_observation_noise_spec,
    observation_noise_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "benchmark" / "build_false_positive_replay_report_issue_3300.py"
)


def _row(
    *,
    route_complete: bool = True,
    collision: bool = False,
    progress: float = 1.0,
    action: str = "go",
    noise: bool = False,
    pedestrians_added: int = 0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "planner_key": "goal",
        "algo": "goal",
        "scenario_id": "planner_sanity_simple",
        "seed": 0,
        "kinematics": "differential_drive",
        "observation_mode": "pedestrians",
        "metrics": {
            "route_complete": route_complete,
            "collision": collision,
            "route_progress": progress,
            "runtime_s": 1.5,
        },
        "selected_action": action,
        "observed_actor_count": 2,
    }
    if noise:
        spec = normalize_observation_noise_spec(
            {
                "profile": "issue_3300_false_positive_actor_injection_v1",
                "seed": 3300,
                "pedestrian_false_positive_prob": 1.0,
                "pedestrian_false_positive_radius_m": 3.0,
                "pedestrian_false_positive_radius": 0.35,
            }
        )
        row.update(
            {
                "observation_noise": spec,
                "observation_noise_hash": observation_noise_hash(spec),
                "observation_noise_stats": {
                    "steps_with_noise": 5 if pedestrians_added else 0,
                    "pedestrians_added": pedestrians_added,
                },
                "observed_actor_count": 2 + pedestrians_added,
            }
        )
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_false_positive_observation_noise_adds_pedestrian_deterministically() -> None:
    """Enabled #3300 profile injects a synthetic pedestrian into compatible observation."""
    spec = normalize_observation_noise_spec(
        {
            "profile": "issue_3300_false_positive_actor_injection_v1",
            "seed": 3300,
            "pedestrian_false_positive_prob": 1.0,
            "pedestrian_false_positive_radius_m": 3.0,
            "pedestrian_false_positive_radius": 0.35,
        }
    )
    obs = {
        "robot": {"position": [1.0, 2.0]},
        "pedestrians": {"positions": [[2.0, 2.0]], "velocities": [[0.0, 0.0]], "count": 1},
    }
    rng_a = make_observation_noise_rng(spec, seed=0, scenario_id="planner_sanity_simple")
    rng_b = make_observation_noise_rng(spec, seed=0, scenario_id="planner_sanity_simple")

    noisy_a, stats_a = apply_observation_noise(obs, spec, rng_a)
    noisy_b, stats_b = apply_observation_noise(obs, spec, rng_b)

    assert stats_a["pedestrians_added"] == 1
    assert stats_a["steps_with_noise"] == 1
    assert noisy_a["pedestrians"]["count"] == 2
    assert noisy_a["pedestrians"]["positions"] == noisy_b["pedestrians"]["positions"]
    assert stats_a == stats_b


def test_incompatible_observation_classifies_blocked_unavailable(tmp_path: Path) -> None:
    """No compatible pedestrian observation must not silently become replay success."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write_jsonl(nominal, [_row()])
    _write_jsonl(perturbed, [_row(noise=True, pedestrians_added=0)])

    report = build_false_positive_replay_report(nominal_jsonl=nominal, perturbed_jsonl=perturbed)

    assert report["classification"]["label"] == CLASS_BLOCKED_UNAVAILABLE
    assert report["injection_summary"]["pedestrians_added"] == 0


def test_report_records_profile_hash_per_episode_deltas_and_observed_class(tmp_path: Path) -> None:
    """Replay report records profile hash, per-episode deltas, and observed changes."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write_jsonl(nominal, [_row(route_complete=True, progress=1.0, action="go")])
    _write_jsonl(
        perturbed,
        [_row(route_complete=False, progress=0.6, action="yield", noise=True, pedestrians_added=3)],
    )

    report = build_false_positive_replay_report(nominal_jsonl=nominal, perturbed_jsonl=perturbed)

    assert report["classification"]["label"] == CLASS_OBSERVED
    assert report["injection_summary"]["pedestrians_added"] == 3
    assert report["injection_summary"]["profiles"] == [
        "issue_3300_false_positive_actor_injection_v1"
    ]
    assert report["injection_summary"]["hashes"]
    episode = report["per_episode_deltas"][0]
    assert episode["pedestrians_added"] == 3
    assert episode["metric_delta"]["route_progress"] == -0.4
    assert "route_complete" in episode["changed_fields"]
    assert "selected_action" in episode["changed_fields"]


def test_classifier_distinguishes_scenario_too_weak_and_trace_only(tmp_path: Path) -> None:
    """Classifier keeps no-effect executable smoke separate from trace diagnostics."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write_jsonl(nominal, [_row()])
    _write_jsonl(perturbed, [_row(noise=True, pedestrians_added=1)])

    report = build_false_positive_replay_report(nominal_jsonl=nominal, perturbed_jsonl=perturbed)
    trace_report = {**report, "replay_mode": "trace_derived"}

    assert report["classification"]["label"] == CLASS_SCENARIO_TOO_WEAK
    assert classify_false_positive_replay(trace_report)["label"] == CLASS_TRACE_ONLY_DIAGNOSTIC


def test_cli_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    """CLI emits the reviewable issue #3300 artifact trio."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "robustness_delta.csv"
    output_md = tmp_path / "false_positive_replay_report.md"
    _write_jsonl(nominal, [_row()])
    _write_jsonl(perturbed, [_row(noise=True, pedestrians_added=1)])

    script = runpy.run_path(str(SCRIPT_PATH))
    exit_code = script["main"](
        [
            "--nominal-jsonl",
            str(nominal),
            "--perturbed-jsonl",
            str(perturbed),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "false_positive_actor_injection_replay.v1"
    assert payload["classification"]["label"] == CLASS_SCENARIO_TOO_WEAK
    assert "pedestrians_added" in output_csv.read_text(encoding="utf-8")
    markdown = output_md.read_text(encoding="utf-8")
    assert "Issue #3300 False-Positive Actor-Injection Replay" in markdown
    assert "No full benchmark campaign" in markdown
