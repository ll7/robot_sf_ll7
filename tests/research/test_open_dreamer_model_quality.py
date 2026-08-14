"""Tests for the fail-closed issue-6318 model-quality gate."""

# evidence-writer-exempt: tests deliberately write isolated synthetic contract fixtures under pytest tmp_path.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    assign_deterministic_split,
    compute_return_to_go,
    write_rl_trajectory_dataset,
)
from robot_sf.research.open_dreamer_model_quality import (
    ModelQualityConfig,
    OpenDreamerQualityError,
    _gate_metrics,
    evaluate_model_quality,
)
from scripts.validation.run_open_dreamer_model_quality import main as run_quality_main


def _seed_for(scenario_id: str, split: str, *, start: int) -> int:
    """Find a deterministic seed for one scenario and split."""
    for seed in range(start, start + 10_000):
        if assign_deterministic_split(scenario_id, seed) == split:
            return seed
    raise AssertionError(f"could not find a seed for {scenario_id!r}/{split!r}")


def _episode(
    scenario_id: str,
    seed: int,
    *,
    episode_index: int,
    terminal: bool = True,
) -> RLTrajectoryEpisode:
    """Build a small finite trajectory with an explicit canonical split."""
    rewards = tuple(0.25 + 0.02 * step for step in range(6))
    observations = []
    robot_states = []
    actions = []
    pedestrians = []
    for step in range(6):
        position = [0.2 * step + 0.01 * episode_index, -0.1 * step]
        robot = {
            "position": position,
            "heading": 0.02 * step,
            "velocity": [0.2, -0.1],
        }
        observations.append({"robot": robot, "pedestrians": []})
        robot_states.append(robot)
        actions.append(
            {
                "linear_velocity": 0.2 + 0.01 * step,
                "angular_velocity": -0.2 + 0.01 * episode_index,
            }
        )
        pedestrians.append([])
    terminated = (False, False, False, False, False, terminal)
    truncated = (False,) * 6
    return RLTrajectoryEpisode(
        dataset_id="issue_6318_quality_fixture",
        episode_id=f"{scenario_id}:{seed}:{episode_index}",
        scenario_id=scenario_id,
        seed=seed,
        source_policy_id="quality_fixture_policy",
        split=assign_deterministic_split(scenario_id, seed),
        observations=tuple(observations),
        actions=tuple(actions),
        rewards=rewards,
        return_to_go=tuple(compute_return_to_go(rewards)),
        terminated=terminated,
        truncated=truncated,
        pedestrians=tuple(pedestrians),
        robot_states=tuple(robot_states),
        provenance={"fixture": "synthetic_contract_only", "episode_index": episode_index},
    )


def _fixture_dataset(tmp_path: Path, *, terminal: bool = True) -> Path:
    train_scenario = "quality_train_scenario"
    holdout_scenario = "quality_holdout_scenario"
    episodes = [
        _episode(
            train_scenario,
            _seed_for(train_scenario, "train", start=100),
            episode_index=0,
            terminal=terminal,
        ),
        _episode(
            train_scenario,
            _seed_for(train_scenario, "train", start=200),
            episode_index=1,
            terminal=terminal,
        ),
        _episode(
            holdout_scenario,
            _seed_for(holdout_scenario, "test", start=300),
            episode_index=2,
            terminal=terminal,
        ),
        _episode(
            holdout_scenario,
            _seed_for(holdout_scenario, "test", start=400),
            episode_index=3,
            terminal=terminal,
        ),
    ]
    path = tmp_path / "quality_fixture.jsonl"
    write_rl_trajectory_dataset(episodes, path)
    return path


def _config(dataset_path: Path) -> ModelQualityConfig:
    return ModelQualityConfig(
        dataset_path=dataset_path,
        latent_dim=5,
        min_train_episodes=2,
        min_holdout_episodes=2,
        min_train_transitions=4,
        min_holdout_transitions=4,
        required_baselines=("persistence", "mlp"),
    )


def test_quality_gate_reports_deterministic_metrics_for_sufficient_fixture(tmp_path: Path) -> None:
    dataset_path = _fixture_dataset(tmp_path)
    config = _config(dataset_path)

    first = evaluate_model_quality(config).to_dict()
    second = evaluate_model_quality(config).to_dict()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] in {"passed", "failed_model_quality"}
    assert first["evidence_boundary"] == "diagnostic_only"
    assert set(first["one_step_metrics"]) == {"model", "persistence", "mlp"}
    assert set(first["one_step_metrics"]["model"]) == {
        "next_observation_rmse",
        "reward_mae",
        "continuation_brier",
    }
    assert first["model"]["fitted"] is True
    assert first["model"]["fit_method"] == "ridge_closed_form"
    assert "trained" not in first["model"]
    assert set(first["gate"]["per_baseline"]["persistence"]["comparisons"]) == {
        "one_step.next_observation_rmse",
        "one_step.reward_mae",
        "one_step.continuation_brier",
        "multi_step.next_observation_rmse",
        "multi_step.reward_mae",
        "multi_step.continuation_brier",
    }
    assert "quality_train_scenario" in first["split_summary"]["train"]["scenario_ids"]
    assert "quality_holdout_scenario" in first["split_summary"]["test"]["scenario_ids"]


def test_quality_gate_rejects_multi_step_head_regression() -> None:
    config = _config(Path("quality_fixture.jsonl"))
    metric_names = ("next_observation_rmse", "reward_mae", "continuation_brier")
    one_step = {
        "model": dict.fromkeys(metric_names, 0.1),
        "persistence": dict.fromkeys(metric_names, 0.2),
        "mlp": dict.fromkeys(metric_names, 0.2),
    }
    multi_step = {
        "model": {
            "next_observation_rmse": 0.1,
            "reward_mae": 0.3,
            "continuation_brier": 0.3,
        },
        "persistence": dict.fromkeys(metric_names, 0.2),
        "mlp": dict.fromkeys(metric_names, 0.2),
    }

    gate = _gate_metrics(config, one_step, multi_step)

    assert gate["passed"] is False
    for baseline in ("persistence", "mlp"):
        comparisons = gate["per_baseline"][baseline]["comparisons"]
        assert comparisons["multi_step.next_observation_rmse"]["passed"] is True
        assert comparisons["multi_step.reward_mae"]["passed"] is False
        assert comparisons["multi_step.continuation_brier"]["passed"] is False


def test_quality_gate_blocks_tiny_committed_preview() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    preview = (
        repo_root
        / "docs/context/evidence/issue_4011_rl_trajectory_dataset_smoke_2026-07-02"
        / "issue_4011_smoke.preview.jsonl"
    )
    report = evaluate_model_quality(
        ModelQualityConfig(
            dataset_path=preview,
            min_train_episodes=2,
            min_holdout_episodes=2,
            min_train_transitions=8,
            min_holdout_transitions=4,
        )
    ).to_dict()

    assert report["status"] == "blocked_insufficient_data"
    assert "test episodes 0 < 2" in report["reason"]
    assert "one_step_metrics" not in report


def test_quality_gate_blocks_split_contract_failure(tmp_path: Path) -> None:
    dataset_path = _fixture_dataset(tmp_path)
    payload = dataset_path.read_text(encoding="utf-8").replace(
        '"split": "test"', '"split": "train"'
    )
    dataset_path.write_text(payload, encoding="utf-8")

    report = evaluate_model_quality(_config(dataset_path)).to_dict()

    assert report["status"] == "blocked_contract"
    assert "adaptation failed closed" in report["reason"]


def test_quality_gate_blocks_without_continuation_class_diversity(tmp_path: Path) -> None:
    report = evaluate_model_quality(_config(_fixture_dataset(tmp_path, terminal=False))).to_dict()

    assert report["status"] == "blocked_insufficient_data"
    assert "continuation targets must contain both" in report["reason"]


def test_quality_config_resolves_dataset_relative_to_config(tmp_path: Path) -> None:
    config_path = tmp_path / "quality.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: open_dreamer_model_quality.v1",
                "dataset_path: fixture.jsonl",
                "action_bounds:",
                "  max_linear_speed: 1.0",
                "  max_angular_speed: 1.0",
                "  min_linear_speed: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = ModelQualityConfig.from_yaml(config_path)

    assert config.dataset_path == (tmp_path / "fixture.jsonl").resolve()
    assert config.required_baselines == ("persistence", "mlp")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latent_dim", "not-an-int"),
        ("min_train_episodes", True),
        ("ridge_alpha", "not-a-float"),
    ],
)
def test_quality_config_rejects_malformed_mapping_values(field: str, value: object) -> None:
    payload = {
        "schema_version": "open_dreamer_model_quality.v1",
        "dataset_path": "fixture.jsonl",
        field: value,
    }

    with pytest.raises(OpenDreamerQualityError, match=field):
        ModelQualityConfig.from_mapping(payload, base_dir=Path("."))


def test_quality_cli_writes_blocked_contract_for_invalid_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "invalid_quality.yaml"
    config_path.write_text(
        "schema_version: open_dreamer_model_quality.v1\n"
        "dataset_path: fixture.jsonl\n"
        "latent_dim: not-an-int\n",
        encoding="utf-8",
    )

    exit_code = run_quality_main(
        ["--config", str(config_path), "--output-dir", str(tmp_path / "report")]
    )

    assert exit_code == 2
    report_path = tmp_path / "report/open_dreamer_model_quality.v1.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_contract"
    assert (
        report["reason"]
        == "quality config invalid: latent_dim must be an integer, got 'not-an-int'"
    )
    assert report["config_path"] == str(config_path.resolve())
    assert '"status": "blocked_contract"' in capsys.readouterr().out
