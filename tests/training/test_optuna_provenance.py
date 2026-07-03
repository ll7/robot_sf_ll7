"""Tests for Optuna expert PPO provenance artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import optuna

from robot_sf.training.optuna_provenance import redact_storage_url, write_study_manifest

if TYPE_CHECKING:
    from pathlib import Path


def test_redact_storage_url_hides_credentials() -> None:
    """Storage credentials must not leak into reviewable provenance."""

    rendered = redact_storage_url("postgresql://user:secret@example.test/optuna")

    assert "secret" not in rendered
    assert "***" in rendered


def test_write_study_manifest_records_hashes_and_trials(tmp_path: Path) -> None:
    """Study provenance writes a study manifest and one trial manifest."""

    base_config = tmp_path / "expert_ppo.yaml"
    base_config.write_text("policy_id: demo\n", encoding="utf-8")
    launcher_config = tmp_path / "launcher.yaml"
    launcher_config.write_text(
        "schema_version: robot_sf.optuna_expert_ppo_launcher.v2\n", encoding="utf-8"
    )
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="demo", storage=storage, direction="maximize")
    study.add_trial(
        optuna.trial.create_trial(
            params={"learning_rate": 1e-4},
            distributions={
                "learning_rate": optuna.distributions.FloatDistribution(1e-5, 3e-4, log=True)
            },
            value=1.0,
            user_attrs={"policy_id": "demo_optuna_000"},
        )
    )

    manifest_path = write_study_manifest(
        output_dir=tmp_path / "provenance",
        study=study,
        storage="postgresql://user:secret@example.test/optuna",
        base_config_path=base_config,
        launcher_config_path=launcher_config,
        search_space={"ppo_hyperparams": {"learning_rate": {"type": "float"}}},
        runtime_bounds={"trials": 1, "seed": 4019},
        git_state={"commit": "abc", "branch": "test", "dirty": False},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["trial_count"] == 1
    assert payload["base_config_sha256"]
    assert "secret" not in payload["storage"]
    assert (tmp_path / "provenance" / "trials" / "trial_000.json").exists()
