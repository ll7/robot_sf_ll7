"""Submit-path checkpoint provisioning tests for issue #4613 (follow-up #4663).

These cover the enforced-staged submit-time gate that the always-on cheap guard from PR #4620 was
missing: a remote-backed but locally-absent checkpoint must fail closed at submit time, the
``submit_safe`` boolean must report whether resolvability is sufficient for ``sbatch``, and the
staging report must be persisted next to the other preflight artifacts. All tests are CPU-only and
network-free; the staged download path is monkeypatched so no real fetch happens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

import robot_sf.benchmark.camera_ready._preflight as preflight_module
import robot_sf.benchmark.campaign_checkpoint_preflight as ckpt_module
from robot_sf.benchmark.camera_ready._config_types import (
    CampaignConfig,
    PlannerSpec,
    SeedPolicy,
)
from robot_sf.benchmark.camera_ready._preflight import (
    _CHECKPOINT_PREFLIGHT_REPORT_NAME,
    prepare_campaign_preflight,
)
from robot_sf.benchmark.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
    check_campaign_arm_checkpoints_preflight,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "preflight_campaign_checkpoints.py"
)
WORKTREE_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess env that pins this worktree's source via PYTHONPATH."""
    import os

    env = {
        "PATH": os.environ["PATH"],
        # Pin the worktree's robot_sf/ package ahead of the shared venv's editable install.
        "PYTHONPATH": str(WORKTREE_ROOT),
        "UV_NO_SYNC": "1",
    }
    if extra:
        env.update(extra)
    return env


def _write_registry(tmp_path: Path, models: list[dict]) -> Path:
    """Write a minimal model registry YAML fixture and return its path."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"version": 1, "models": models}, sort_keys=False),
        encoding="utf-8",
    )
    return registry_path


def _write_algo_config(tmp_path: Path, name: str, payload: dict) -> Path:
    """Write an arm algo_config YAML and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _campaign(planners: tuple[PlannerSpec, ...], *, tmp_path: Path) -> CampaignConfig:
    """Build a minimal campaign config wrapping the given planner arms."""
    scenario_path = tmp_path / "scenarios.yaml"
    if not scenario_path.exists():
        scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    return CampaignConfig(
        name="checkpoint_submit_preflight_test",
        scenario_matrix_path=scenario_path,
        planners=planners,
        seed_policy=SeedPolicy(),
    )


# --- submit_safe semantics ------------------------------------------------


def test_submit_safe_false_for_stageable_remote_in_metadata_only(tmp_path: Path) -> None:
    """metadata_only passes resolvability for stageable_remote but reports submit_safe=False."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "remote",
                "local_path": "output/model_cache/remote/model.zip",  # absent locally
                "github_release": {
                    "asset_name": "remote-model.zip",
                    "url": "https://example.invalid/remote-model.zip",
                    "sha256": "0" * 64,
                },
            }
        ],
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "remote"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    summary = check_campaign_arm_checkpoints_preflight(cfg, registry_path=registry_path)
    assert summary["arms"][0]["status"] == "stageable_remote"
    assert summary["submit_safe"] is False
    assert summary["stage"] is False


def test_submit_safe_true_for_present_local_in_metadata_only(tmp_path: Path) -> None:
    """A present_local checkpoint under metadata_only is submit-safe."""
    local_model = tmp_path / "weights" / "model.zip"
    local_model.parent.mkdir(parents=True)
    local_model.write_text("checkpoint", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path, [{"model_id": "present", "local_path": str(local_model)}]
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "present"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    summary = check_campaign_arm_checkpoints_preflight(cfg, registry_path=registry_path)
    assert summary["arms"][0]["status"] == "present_local"
    assert summary["submit_safe"] is True


def test_submit_safe_true_after_staged(tmp_path: Path) -> None:
    """Under stage=True, a staged checkpoint is submit-safe (no real download needed)."""
    local_model = tmp_path / "weights" / "model.zip"
    local_model.parent.mkdir(parents=True)
    local_model.write_text("checkpoint", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path, [{"model_id": "present", "local_path": str(local_model)}]
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "present"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    summary = check_campaign_arm_checkpoints_preflight(cfg, stage=True, registry_path=registry_path)
    assert summary["stage"] is True
    assert summary["arms"][0]["status"] == "staged"
    assert summary["submit_safe"] is True


def test_submit_safe_true_for_empty_campaign(tmp_path: Path) -> None:
    """A campaign with no checkpoint-bearing arms is trivially submit-safe."""
    cfg = _campaign((PlannerSpec(key="orca", algo="orca"),), tmp_path=tmp_path)
    summary = check_campaign_arm_checkpoints_preflight(cfg)
    assert summary["submit_safe"] is True


# --- enforced_staged mode wires stage=True via prepare_campaign_preflight --


def test_prepare_enforced_staged_invokes_stage_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """enforced_staged mode must call resolve_model_path (stage=True), not the cheap guard."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "remote",
                "local_path": str(tmp_path / "staged" / "remote" / "model.zip"),
                "github_release": {
                    "asset_name": "remote-model.zip",
                    "url": "https://example.invalid/remote-model.zip",
                    "sha256": "0" * 64,
                },
            }
        ],
    )
    staged_path = tmp_path / "staged" / "remote" / "model.zip"
    staged_path.parent.mkdir(parents=True)
    staged_path.write_text("staged", encoding="utf-8")

    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "remote"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )

    # Bypass real scenario loading so this provisioning-only test does not pin the scenario format.
    monkeypatch.setattr(preflight_module, "_load_campaign_scenarios", lambda *_a, **_k: [])

    calls: list[dict[str, object]] = []

    def fake_resolve_model_path(
        model_id: str,
        *,
        registry_path: str | Path | None = None,
        allow_download: bool = False,
        cache_dir: str | Path | None = None,
    ) -> Path:
        calls.append(
            {
                "model_id": model_id,
                "registry_path": registry_path,
                "allow_download": allow_download,
                "cache_dir": cache_dir,
            }
        )
        return staged_path

    monkeypatch.setattr(ckpt_module, "resolve_model_path", fake_resolve_model_path)

    prepared = prepare_campaign_preflight(
        cfg,
        output_root=tmp_path / "out",
        label="submit",
        checkpoint_preflight_mode="enforced_staged",
        checkpoint_registry_path=registry_path,
        checkpoint_cache_dir=tmp_path / "cache",
    )

    assert len(calls) == 1
    assert calls[0]["allow_download"] is True
    assert calls[0]["cache_dir"] == tmp_path / "cache"
    summary = prepared["checkpoint_preflight_summary"]
    assert summary["stage"] is True
    assert summary["submit_safe"] is True
    assert summary["arms"][0]["status"] == "staged"
    report_path = Path(prepared["checkpoint_preflight_report_path"])
    assert report_path.name == _CHECKPOINT_PREFLIGHT_REPORT_NAME["enforced_staged"]
    assert report_path.is_file()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["mode"] == "enforced_staged"
    assert persisted["stage"] is True
    assert persisted["submit_safe"] is True
    # Manifest surfaces the provisioning artifact path.
    assert "preflight_checkpoint_provisioning" in prepared["manifest_payload"]["artifacts"]


def test_prepare_metadata_only_writes_resolvability_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """metadata_only mode persists checkpoint_resolvability.json with submit_safe=False."""
    local_model = tmp_path / "weights" / "model.zip"
    local_model.parent.mkdir(parents=True)
    local_model.write_text("checkpoint", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path, [{"model_id": "present", "local_path": str(local_model)}]
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "present"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    monkeypatch.setattr(preflight_module, "_load_campaign_scenarios", lambda *_a, **_k: [])
    prepared = prepare_campaign_preflight(
        cfg,
        output_root=tmp_path / "out",
        label="meta",
        checkpoint_registry_path=registry_path,
    )
    report_path = Path(prepared["checkpoint_preflight_report_path"])
    assert report_path.name == _CHECKPOINT_PREFLIGHT_REPORT_NAME["metadata_only"]
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["mode"] == "metadata_only"
    assert persisted["stage"] is False
    assert persisted["submit_safe"] is True  # present_local only


def test_prepare_enforced_staged_fails_closed_when_staging_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """enforced_staged raises before scenarios load when staging cannot materialize the file."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "remote",
                "github_release": {"asset_name": "x", "url": "u", "sha256": "0" * 64},
            }
        ],
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "remote"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )

    def boom(*args, **kwargs):
        raise FileNotFoundError("remote unreachable")

    monkeypatch.setattr(ckpt_module, "resolve_model_path", boom)
    with pytest.raises(CampaignCheckpointPreflightError, match="stage_failed"):
        prepare_campaign_preflight(
            cfg,
            output_root=tmp_path / "out",
            label="fail",
            checkpoint_preflight_mode="enforced_staged",
            checkpoint_registry_path=registry_path,
        )


# --- CLI: submit_safe + --report-path --------------------------------------


def test_cli_json_reports_submit_safe_false_for_stageable_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI --json output surfaces submit_safe=false for an unstaged remote-only arm."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "remote",
                "local_path": "output/model_cache/remote/model.zip",
                "github_release": {
                    "asset_name": "remote-model.zip",
                    "url": "https://example.invalid/remote-model.zip",
                    "sha256": "0" * 64,
                },
            }
        ],
    )
    config_path = tmp_path / "campaign.yaml"
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "remote"})
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli_test_campaign",
                "scenario_matrix": str(scenario_path),
                "planners": [{"key": "ppo", "algo": "ppo", "algo_config": str(algo_config)}],
                "seed_policy": {"mode": "explicit", "seed_set": "paper_eval_s20"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "submit_packet" / "checkpoint_resolvability.json"

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT_PATH,
            "--config",
            str(config_path),
            "--registry-path",
            str(registry_path),
            "--json",
            "--report-path",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=_subprocess_env({"ROBOT_SF_TESTING": "1"}),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["submit_safe"] is False
    assert payload["arms"][0]["status"] == "stageable_remote"
    assert report_path.is_file()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["submit_safe"] is False


def test_cli_stage_writes_report_and_exits_nonzero_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The --stage CLI writes a fail-closed report and exits 3 when staging cannot resolve."""
    registry_path = _write_registry(tmp_path, [])
    config_path = tmp_path / "campaign.yaml"
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    algo_config = _write_algo_config(
        tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "definitely_absent"}
    )
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "cli_stage_fail_campaign",
                "scenario_matrix": str(scenario_path),
                "planners": [{"key": "ppo", "algo": "ppo", "algo_config": str(algo_config)}],
                "seed_policy": {"mode": "explicit", "seed_set": "paper_eval_s20"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "blocked_report.json"

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            SCRIPT_PATH,
            "--config",
            str(config_path),
            "--registry-path",
            str(registry_path),
            "--stage",
            "--report-path",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=_subprocess_env(),
        check=False,
    )
    assert result.returncode == 3, result.stderr
    assert report_path.is_file()
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "blocked"
    assert "ppo" in persisted["arms"]
