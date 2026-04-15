"""Tests for the SAC autoresearch experiment harness."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.tools import sac_autoresearch as mod


def test_format_result_row_prefers_total_episodes() -> None:
    """Result rows should prefer the evaluation report's total_episodes field."""

    row = mod._format_result_row(
        {"name": "exp", "config": "cfg", "description": "desc"},
        {
            "success_rate": 0.8,
            "mean_min_distance": 3.9,
            "mean_avg_speed": 1.96,
            "gate_pass": True,
            "episodes": 999,
            "total_episodes": 30,
            "duration_s": "12.0",
        },
    )

    assert row[7] == "30"


def test_main_uses_config_eval_settings_and_writes_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The harness should honor per-config eval settings and still record results."""

    experiments_path = tmp_path / "experiments.yaml"
    config_path = tmp_path / "train.yaml"
    scenario_matrix = tmp_path / "subset.yaml"
    algo_config = tmp_path / "algo.yaml"
    output_dir = tmp_path / "models"

    scenario_matrix.write_text("scenarios: []\n", encoding="utf-8")
    algo_config.write_text("model_path: model.zip\n", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "policy_id: sac_test_policy",
                "scenario_config: subset.yaml",
                "total_timesteps: 1000",
                f"output_dir: {output_dir}",
                "evaluation:",
                "  scenario_matrix: subset.yaml",
                "  algo_config: algo.yaml",
                "  workers: 2",
                "  horizon: 250",
                "  dt: 0.2",
                "  min_success_rate: 0.4",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    experiments_path.write_text(
        "- name: exp\n  config: train.yaml\n  description: test experiment\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(mod, "RESULTS_PATH", (tmp_path / "results" / "results.tsv"))

    commands: list[list[str]] = []

    def _fake_run_process(command: list[str], *, allow_failure: bool = False) -> int:
        commands.append(command)
        if any("train_sac_sb3.py" in part for part in command):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "sac_test_policy.zip").write_text("checkpoint", encoding="utf-8")
            return 0
        if any("evaluate_sac.py" in part for part in command):
            eval_output_dir = Path(command[command.index("--output-dir") + 1])
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            (eval_output_dir / "sac_eval_summary.json").write_text(
                json.dumps(
                    {
                        "success_rate": 0.8,
                        "mean_min_distance": 3.9,
                        "mean_avg_speed": 1.96,
                        "gate_pass": True,
                        "total_episodes": 30,
                    }
                ),
                encoding="utf-8",
            )
            return 1 if allow_failure else 0
        return 0

    monkeypatch.setattr(mod, "_run_process", _fake_run_process)

    exit_code = mod.main(["--experiments", str(experiments_path)])

    assert exit_code == 0
    assert len(commands) == 2

    eval_command = commands[1]
    assert eval_command[0:4] == ["uv", "run", "python", "scripts/validation/evaluate_sac.py"]
    assert eval_command[eval_command.index("--scenario-matrix") + 1] == str(
        scenario_matrix.resolve()
    )
    assert eval_command[eval_command.index("--algo-config") + 1] == str(algo_config.resolve())
    assert eval_command[eval_command.index("--workers") + 1] == "2"
    assert eval_command[eval_command.index("--horizon") + 1] == "250"
    assert eval_command[eval_command.index("--dt") + 1] == "0.2"
    assert eval_command[eval_command.index("--min-success-rate") + 1] == "0.4"
    assert eval_command[eval_command.index("--device") + 1] == "cpu"

    with mod.RESULTS_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))

    assert rows[0][0] == "name"
    assert rows[1][0] == "exp"
    assert rows[1][3] == "0.8000"
    assert rows[1][7] == "30"
