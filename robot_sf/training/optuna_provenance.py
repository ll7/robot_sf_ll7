"""Provenance artifact helpers for Optuna expert PPO studies."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import optuna
import yaml
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


def hash_file(path: Path | None) -> str | None:
    """Return a SHA-256 digest for ``path`` or ``None`` when no file is supplied."""

    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(payload: object) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable payload."""

    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def redact_storage_url(storage: str) -> str:
    """Render a storage URL with credentials hidden when possible.

    Returns:
        Storage URL string with password credentials redacted when parsing succeeds.
    """

    try:
        return make_url(storage).render_as_string(hide_password=True)
    except (ArgumentError, TypeError, ValueError):
        return storage


def git_snapshot(repo_root: Path | None = None) -> dict[str, object]:
    """Collect compact Git state for provenance manifests.

    Returns:
        Mapping with commit, branch, and dirty-tree status when available.
    """

    cwd = repo_root or Path.cwd()
    snapshot: dict[str, object] = {"commit": None, "branch": None, "dirty": None}
    try:
        snapshot["commit"] = _git(cwd, "rev-parse", "HEAD")
        snapshot["branch"] = _git(cwd, "branch", "--show-current")
        snapshot["dirty"] = bool(_git(cwd, "status", "--porcelain"))
    except (FileNotFoundError, subprocess.CalledProcessError):
        snapshot["dirty"] = None
    return snapshot


def write_trial_manifest(  # noqa: PLR0913
    *,
    output_dir: Path,
    trial: optuna.trial.FrozenTrial,
    study_name: str,
    storage: str,
    base_config_path: Path,
    launcher_config_path: Path | None,
    search_space: object | None,
    runtime_bounds: dict[str, object],
    git_state: dict[str, object] | None = None,
) -> Path:
    """Write a reviewable JSON manifest for one frozen Optuna trial.

    Returns:
        Path to the written trial manifest.
    """

    trial_dir = output_dir / "trials"
    trial_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "robot_sf.optuna_expert_ppo_trial_provenance.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "study_name": study_name,
        "storage": redact_storage_url(storage),
        "trial_number": trial.number,
        "trial_state": trial.state.name,
        "objective_value": trial.value,
        "params": dict(trial.params),
        "user_attrs": _jsonable_mapping(trial.user_attrs),
        "base_config_path": str(base_config_path),
        "base_config_sha256": hash_file(base_config_path),
        "launcher_config_path": str(launcher_config_path) if launcher_config_path else None,
        "launcher_config_sha256": hash_file(launcher_config_path) if launcher_config_path else None,
        "search_space_sha256": hash_payload(search_space) if search_space is not None else None,
        "runtime_bounds": runtime_bounds,
        "git": git_state or git_snapshot(),
    }
    output_path = trial_dir / f"trial_{trial.number:03d}.json"
    _write_json(output_path, payload)
    return output_path


def write_study_manifest(
    *,
    output_dir: Path,
    study: optuna.study.Study,
    storage: str,
    base_config_path: Path,
    launcher_config_path: Path | None,
    search_space: object | None,
    runtime_bounds: dict[str, object],
    git_state: dict[str, object] | None = None,
) -> Path:
    """Write study-level provenance and per-trial manifests.

    Returns:
        Path to the written study manifest.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    git = git_state or git_snapshot()
    trial_paths = [
        write_trial_manifest(
            output_dir=output_dir,
            trial=trial,
            study_name=study.study_name,
            storage=storage,
            base_config_path=base_config_path,
            launcher_config_path=launcher_config_path,
            search_space=search_space,
            runtime_bounds=runtime_bounds,
            git_state=git,
        )
        for trial in study.trials
    ]
    payload = {
        "schema_version": "robot_sf.optuna_expert_ppo_study_provenance.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "study_name": study.study_name,
        "direction": study.direction.name,
        "storage": redact_storage_url(storage),
        "trial_count": len(study.trials),
        "completed_trial_count": sum(
            1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ),
        "base_config_path": str(base_config_path),
        "base_config_sha256": hash_file(base_config_path),
        "launcher_config_path": str(launcher_config_path) if launcher_config_path else None,
        "launcher_config_sha256": hash_file(launcher_config_path) if launcher_config_path else None,
        "search_space": search_space,
        "search_space_sha256": hash_payload(search_space) if search_space is not None else None,
        "runtime_bounds": runtime_bounds,
        "git": git,
        "trial_manifests": [str(path) for path in trial_paths],
    }
    output_path = output_dir / "study_manifest.json"
    _write_json(output_path, payload)
    return output_path


def write_best_config_artifacts(
    *,
    best_trial_path: Path,
    best_config_path: Path,
    report_path: Path,
    trial: optuna.trial.FrozenTrial,
    effective_config: dict[str, object],
    selection: dict[str, object],
) -> None:
    """Write best-trial JSON, best-config YAML, and a compact selection report."""

    _write_json(best_trial_path, {"trial": _trial_payload(trial), "selection": selection})
    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    best_config_path.write_text(yaml.safe_dump(effective_config, sort_keys=False), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_selection_report(trial=trial, selection=selection), encoding="utf-8")


def _trial_payload(trial: optuna.trial.FrozenTrial) -> dict[str, object]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "value": trial.value,
        "params": dict(trial.params),
        "user_attrs": _jsonable_mapping(trial.user_attrs),
    }


def _selection_report(*, trial: optuna.trial.FrozenTrial, selection: dict[str, object]) -> str:
    lines = [
        "# Optuna Best-Config Selection",
        "",
        f"- study_name: `{selection['study_name']}`",
        f"- direction: `{selection['direction']}`",
        f"- selected_trial: `{trial.number}`",
        f"- selected_value: `{trial.value}`",
        f"- complete_trials_considered: `{selection['complete_trials_considered']}`",
        f"- safety_filter_applied: `{selection['safety_filter_applied']}`",
        f"- base_config_sha256: `{selection['base_config_sha256']}`",
        "",
        "This report records harness selection only; it is not benchmark evidence.",
        "",
    ]
    return "\n".join(lines)


def _jsonable_mapping(payload: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(payload, default=str, sort_keys=True))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()
