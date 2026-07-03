"""Tests for Optuna best-config export tooling."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import optuna
import yaml

from scripts.tools.export_optuna_best_config import main

if TYPE_CHECKING:
    from pathlib import Path


def test_export_optuna_best_config_prefers_safety_feasible_trial(tmp_path: Path) -> None:
    """Exporter reconstructs a config from the selected complete feasible trial."""

    db_path = tmp_path / "study.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(study_name="demo", storage=storage, direction="maximize")
    distribution = optuna.distributions.FloatDistribution(1e-5, 3e-4, log=True)
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 2e-4},
            distributions={"learning_rate": distribution},
            value=10.0,
            user_attrs={"safety_constraints_feasible": False},
        )
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 1e-4},
            distributions={"learning_rate": distribution},
            value=5.0,
            user_attrs={"safety_constraints_feasible": True},
        )
    )
    base_config = tmp_path / "expert_ppo.yaml"
    base_config.write_text(
        yaml.safe_dump({"policy_id": "demo", "ppo_hyperparams": {"n_epochs": 5}}),
        encoding="utf-8",
    )
    out_path = tmp_path / "best_config.yaml"
    report_path = tmp_path / "selection_report.md"
    best_trial_path = tmp_path / "best_trial.json"

    exit_code = main(
        [
            "--db",
            str(db_path),
            "--study-name",
            "demo",
            "--base-config",
            str(base_config),
            "--out",
            str(out_path),
            "--best-trial-json",
            str(best_trial_path),
            "--report",
            str(report_path),
        ]
    )

    assert exit_code == 0
    rendered = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert rendered["ppo_hyperparams"]["n_epochs"] == 5
    assert rendered["ppo_hyperparams"]["learning_rate"] == 1e-4
    best_trial = json.loads(best_trial_path.read_text(encoding="utf-8"))
    assert best_trial["trial"]["number"] == 1
    assert "not benchmark evidence" in report_path.read_text(encoding="utf-8")
