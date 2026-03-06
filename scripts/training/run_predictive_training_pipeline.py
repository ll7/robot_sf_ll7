#!/usr/bin/env python3
"""Run config-driven predictive planner training and final evaluation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from loguru import logger

from robot_sf.training.scenario_loader import load_scenarios

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONTRACT_VERSION = "benchmark-reset-v2"
_TRAINING_FAMILY = "prediction_planner"


@dataclass
class PipelinePaths:
    """Resolved output paths for one pipeline run."""

    root: Path
    base_dataset: Path
    hardcase_dataset: Path
    mixed_dataset: Path
    train_dir: Path
    checkpoint: Path
    eval_dir: Path
    diagnostics_dir: Path
    campaign_dir: Path
    final_summary: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read YAML from ``path`` and return a mapping; raise ``TypeError`` for non-mappings."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return payload


def _resolve(path_raw: str | Path, *, base: Path) -> Path:
    """Resolve ``path_raw`` against ``base`` and return an absolute ``Path``."""
    path = Path(path_raw)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _run(cmd: list[str], *, log_level: str = "INFO") -> None:
    logger.info("Running: {}", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("LOGURU_LEVEL", str(log_level).upper())
    subprocess.run(cmd, check=True, env=env, cwd=_REPO_ROOT)


def _git_hash() -> str:
    """Return current git commit hash when available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=_REPO_ROOT).strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def _sha1_file(path: Path) -> str:
    """Return SHA1 hash for a file."""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON mapping from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _dataset_npz_diagnostics(path: Path) -> dict[str, Any]:
    """Compute dataset-level diagnostics from a predictive rollout NPZ."""
    raw = np.load(path)
    state = np.asarray(raw["state"], dtype=np.float32)
    target = np.asarray(raw["target"], dtype=np.float32)
    mask = np.asarray(raw["mask"], dtype=np.float32)
    target_mask = (
        np.asarray(raw["target_mask"], dtype=np.float32)
        if "target_mask" in raw
        else np.repeat(mask[:, :, None], target.shape[2], axis=2).astype(np.float32)
    )
    agent_rows = np.sum(mask, axis=1)
    target_rows = np.sum(target_mask, axis=(1, 2))
    return {
        "num_samples": int(state.shape[0]),
        "max_agents": int(state.shape[1]),
        "horizon_steps": int(target.shape[2]),
        "active_agent_ratio": float(np.mean(mask)),
        "active_target_ratio": float(np.mean(target_mask)),
        "invalid_mask_count": int(np.count_nonzero(mask <= 0.0)),
        "invalid_target_mask_count": int(np.count_nonzero(target_mask <= 0.0)),
        "empty_agent_rows": int(np.count_nonzero(agent_rows <= 0.0)),
        "empty_target_rows": int(np.count_nonzero(target_rows <= 0.0)),
    }


def _write_dataset_manifest(
    *,
    dataset_path: Path,
    summary_path: Path,
    role: str,
    run_id: str,
    config_path: Path,
    config_hash: str,
    git_commit: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a reproducible dataset manifest next to a generated dataset."""
    summary = _read_json(summary_path)
    manifest = {
        "dataset_id": f"{run_id}:{role}",
        "contract_version": _CONTRACT_VERSION,
        "training_family": _TRAINING_FAMILY,
        "artifact_role": role,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "git_commit": git_commit,
        "dataset_path": str(dataset_path),
        "dataset_sha1": _sha1_file(dataset_path),
        "summary_path": str(summary_path),
        "summary": summary,
        "diagnostics": _dataset_npz_diagnostics(dataset_path),
        "extra": extra or {},
    }
    manifest_path = dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    return manifest_path


def _run_capture_json(
    cmd: list[str],
    *,
    log_level: str = "INFO",
    allow_failure: bool = False,
) -> dict[str, Any]:
    logger.info("Running: {}", " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("LOGURU_LEVEL", str(log_level).upper())
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=_REPO_ROOT,
    )
    if result.returncode != 0:
        if not allow_failure:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )
        return {
            "status": "failed",
            "return_code": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    out = (result.stdout or "").strip()
    if not out:
        return {"status": "ok", "return_code": 0}
    try:
        payload = json.loads(out)
        if isinstance(payload, dict):
            payload.setdefault("status", "ok")
            payload.setdefault("return_code", 0)
            return payload
        return {"status": "ok", "return_code": 0, "payload": payload}
    except json.JSONDecodeError:
        logger.warning("Command did not emit JSON payload; ignoring stdout")
        return {"status": "ok", "return_code": 0, "stdout": out}


def _build_random_seed_manifest(
    *,
    scenario_matrix: Path,
    seeds_per_scenario: int,
    random_seed_base: int,
    output_path: Path,
) -> Path:
    """Create random seed manifest over all scenarios for base data collection."""
    if seeds_per_scenario < 1:
        raise ValueError("base_collection.seeds_per_scenario must be >= 1")
    scenarios = load_scenarios(scenario_matrix)
    manifest: dict[str, list[int]] = {}
    for idx, scenario in enumerate(scenarios):
        scenario_id = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
        )
        if scenario_id == "unknown":
            continue
        base = int(random_seed_base + (idx * 100_000))
        manifest[scenario_id] = [base + i for i in range(seeds_per_scenario)]
    if not manifest:
        raise RuntimeError(f"Could not build random seed manifest from {scenario_matrix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")
    return output_path


def _paths_from_config(
    cfg: dict[str, Any],
    *,
    run_id: str,
    base_dir: Path,
    output_base_dir: Path,
) -> PipelinePaths:
    out_cfg = cfg.get("output", {})
    if not isinstance(out_cfg, dict):
        raise TypeError("output must be a mapping")
    root = _resolve(
        out_cfg.get("root", "output/tmp/predictive_planner/pipeline"),
        base=output_base_dir,
    )
    run_root = root / run_id
    datasets_dir = run_root / "datasets"
    train_dir = run_root / "training"
    eval_dir = run_root / "eval"
    diagnostics_dir = run_root / "diagnostics"
    campaign_dir = run_root / "campaign"
    return PipelinePaths(
        root=run_root,
        base_dataset=datasets_dir / "predictive_rollouts_base.npz",
        hardcase_dataset=datasets_dir / "predictive_rollouts_hardcase.npz",
        mixed_dataset=datasets_dir / "predictive_rollouts_mixed.npz",
        train_dir=train_dir,
        checkpoint=train_dir / "predictive_model.pt",
        eval_dir=eval_dir,
        diagnostics_dir=diagnostics_dir,
        campaign_dir=campaign_dir,
        final_summary=run_root / "final_performance_summary.json",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Predictive pipeline YAML config path.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Optional run id override. Defaults to config.experiment.run_id + timestamp.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Execute full predictive training pipeline from one YAML config."""
    args = _parse_args()
    logger.remove()
    logger.add(sys.stderr, level=str(args.log_level).upper())

    config_path = args.config.resolve()
    cfg = _read_yaml(config_path)
    config_text = config_path.read_text(encoding="utf-8")
    config_hash = hashlib.sha1(config_text.encode("utf-8")).hexdigest()
    git_commit = _git_hash()
    base_dir = config_path.parent.resolve()

    exp_cfg = cfg.get("experiment", {})
    if not isinstance(exp_cfg, dict):
        raise TypeError("experiment must be a mapping")
    run_prefix = str(exp_cfg.get("run_id", "predictive_br07_all_maps"))
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{run_prefix}_{run_stamp}"

    # Keep scenario/config references config-relative, but write outputs under repo cwd.
    paths = _paths_from_config(
        cfg,
        run_id=run_id,
        base_dir=base_dir,
        output_base_dir=Path.cwd().resolve(),
    )
    paths.root.mkdir(parents=True, exist_ok=True)
    logger.info("Pipeline run root: {}", paths.root)

    scenario_cfg = cfg.get("scenarios", {})
    if not isinstance(scenario_cfg, dict):
        raise TypeError("scenarios must be a mapping")
    scenario_matrix = _resolve(
        scenario_cfg.get("scenario_matrix", "configs/scenarios/classic_interactions.yaml"),
        base=base_dir,
    )
    hard_seed_manifest = _resolve(
        scenario_cfg.get("hard_seed_manifest", "configs/benchmarks/predictive_hard_seeds_v1.yaml"),
        base=base_dir,
    )
    planner_grid = _resolve(
        scenario_cfg.get(
            "planner_grid", "configs/benchmarks/predictive_sweep_planner_grid_v1.yaml"
        ),
        base=base_dir,
    )

    base_collection = cfg.get("base_collection", {})
    if not isinstance(base_collection, dict):
        raise TypeError("base_collection must be a mapping")
    base_manifest = paths.root / "base_random_seed_manifest.yaml"
    _build_random_seed_manifest(
        scenario_matrix=scenario_matrix,
        seeds_per_scenario=int(base_collection.get("seeds_per_scenario", 4)),
        random_seed_base=int(base_collection.get("random_seed_base", 11_000)),
        output_path=base_manifest,
    )

    # 1) Collect base dataset over all scenarios with randomized seed manifest.
    _run(
        [
            sys.executable,
            "scripts/training/collect_predictive_hardcase_data.py",
            "--scenario-matrix",
            str(scenario_matrix),
            "--seed-manifest",
            str(base_manifest),
            "--max-steps",
            str(int(base_collection.get("max_steps", 200))),
            "--max-agents",
            str(int(base_collection.get("max_agents", 24))),
            "--horizon-steps",
            str(int(base_collection.get("horizon_steps", 8))),
            "--max-speed",
            str(float(base_collection.get("max_speed", 1.2))),
            "--output",
            str(paths.base_dataset),
        ],
        log_level=args.log_level,
    )
    base_dataset_manifest = _write_dataset_manifest(
        dataset_path=paths.base_dataset,
        summary_path=paths.base_dataset.with_suffix(".json"),
        role="predictive_base_dataset",
        run_id=run_id,
        config_path=config_path,
        config_hash=config_hash,
        git_commit=git_commit,
        extra={"seed_manifest": str(base_manifest)},
    )

    # 2) Collect hardcase dataset from fixed manifest.
    hardcase_cfg = cfg.get("hardcase_collection", {})
    if not isinstance(hardcase_cfg, dict):
        raise TypeError("hardcase_collection must be a mapping")
    _run(
        [
            sys.executable,
            "scripts/training/collect_predictive_hardcase_data.py",
            "--scenario-matrix",
            str(scenario_matrix),
            "--seed-manifest",
            str(hard_seed_manifest),
            "--max-steps",
            str(int(hardcase_cfg.get("max_steps", 220))),
            "--max-agents",
            str(int(hardcase_cfg.get("max_agents", 24))),
            "--horizon-steps",
            str(int(hardcase_cfg.get("horizon_steps", 8))),
            "--max-speed",
            str(float(hardcase_cfg.get("max_speed", 1.2))),
            "--output",
            str(paths.hardcase_dataset),
        ],
        log_level=args.log_level,
    )
    hardcase_dataset_manifest = _write_dataset_manifest(
        dataset_path=paths.hardcase_dataset,
        summary_path=paths.hardcase_dataset.with_suffix(".json"),
        role="predictive_hardcase_dataset",
        run_id=run_id,
        config_path=config_path,
        config_hash=config_hash,
        git_commit=git_commit,
        extra={"seed_manifest": str(hard_seed_manifest)},
    )

    # 3) Build mixed dataset.
    mixing_cfg = cfg.get("mixing", {})
    if not isinstance(mixing_cfg, dict):
        raise TypeError("mixing must be a mapping")
    _run(
        [
            sys.executable,
            "scripts/training/build_predictive_mixed_dataset.py",
            "--base-dataset",
            str(paths.base_dataset),
            "--hardcase-dataset",
            str(paths.hardcase_dataset),
            "--hardcase-repeat",
            str(int(mixing_cfg.get("hardcase_repeat", 2))),
            "--shuffle-seed",
            str(int(mixing_cfg.get("shuffle_seed", 42))),
            "--output",
            str(paths.mixed_dataset),
        ],
        log_level=args.log_level,
    )
    mixed_dataset_manifest = _write_dataset_manifest(
        dataset_path=paths.mixed_dataset,
        summary_path=paths.mixed_dataset.with_suffix(".json"),
        role="predictive_mixed_dataset",
        run_id=run_id,
        config_path=config_path,
        config_hash=config_hash,
        git_commit=git_commit,
        extra={
            "base_dataset": str(paths.base_dataset),
            "hardcase_dataset": str(paths.hardcase_dataset),
        },
    )

    # 4) Train with proxy + W&B.
    train_cfg = cfg.get("training", {})
    if not isinstance(train_cfg, dict):
        raise TypeError("training must be a mapping")
    wandb_cfg = cfg.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        raise TypeError("wandb must be a mapping")

    train_cmd = [
        sys.executable,
        "scripts/training/train_predictive_planner.py",
        "--dataset",
        str(paths.mixed_dataset),
        "--output-dir",
        str(paths.train_dir),
        "--model-id",
        str(train_cfg.get("model_id", f"predictive_proxy_selected_v2_{run_stamp}")),
        "--epochs",
        str(int(train_cfg.get("epochs", 40))),
        "--batch-size",
        str(int(train_cfg.get("batch_size", 128))),
        "--lr",
        str(float(train_cfg.get("lr", 3e-4))),
        "--weight-decay",
        str(float(train_cfg.get("weight_decay", 1e-5))),
        "--val-split",
        str(float(train_cfg.get("val_split", 0.2))),
        "--seed",
        str(int(train_cfg.get("seed", 42))),
        "--hidden-dim",
        str(int(train_cfg.get("hidden_dim", 128))),
        "--message-passing-steps",
        str(int(train_cfg.get("message_passing_steps", 3))),
        "--max-val-ade",
        str(float(train_cfg.get("max_val_ade", 1.2))),
        "--max-val-fde",
        str(float(train_cfg.get("max_val_fde", 2.0))),
        "--proxy-scenario-matrix",
        str(scenario_matrix),
        "--proxy-seed-manifest",
        str(hard_seed_manifest),
        "--proxy-every-epochs",
        str(int(train_cfg.get("proxy_every_epochs", 5))),
        "--proxy-horizon",
        str(int(train_cfg.get("proxy_horizon", 120))),
        "--proxy-dt",
        str(float(train_cfg.get("proxy_dt", 0.1))),
        "--proxy-workers",
        str(int(train_cfg.get("proxy_workers", 1))),
        "--select-by-proxy",
    ]
    if bool(wandb_cfg.get("enabled", True)):
        train_cmd.extend(
            [
                "--wandb-enabled",
                "--wandb-project",
                str(wandb_cfg.get("project", "robot_sf")),
                "--wandb-entity",
                str(wandb_cfg.get("entity", "ll7")),
                "--wandb-group",
                str(wandb_cfg.get("group", "predictive-br07")),
                "--wandb-job-type",
                str(wandb_cfg.get("job_type", "train")),
                "--wandb-name",
                str(wandb_cfg.get("name", run_id)),
                "--wandb-mode",
                str(wandb_cfg.get("mode", "online")),
            ]
        )
        tags = wandb_cfg.get("tags", ["predictive", "br07", "all-maps", "randomized-seeds"])
        if isinstance(tags, list):
            train_cmd.append("--wandb-tags")
            train_cmd.extend([str(tag) for tag in tags])

    _run(train_cmd, log_level=args.log_level)
    training_summary_path = paths.train_dir / "training_summary.json"
    training_summary = _read_json(training_summary_path)

    # 5) Final evaluations and campaign summary.
    eval_cfg = cfg.get("evaluation", {})
    if not isinstance(eval_cfg, dict):
        raise TypeError("evaluation must be a mapping")

    eval_payload = _run_capture_json(
        [
            sys.executable,
            "scripts/validation/evaluate_predictive_planner.py",
            "--checkpoint",
            str(paths.checkpoint),
            "--scenario-matrix",
            str(scenario_matrix),
            "--workers",
            str(int(eval_cfg.get("workers", 1))),
            "--horizon",
            str(int(eval_cfg.get("horizon", 120))),
            "--dt",
            str(float(eval_cfg.get("dt", 0.1))),
            "--output-dir",
            str(paths.eval_dir),
            "--tag",
            "final_eval",
        ],
        log_level=args.log_level,
        allow_failure=True,
    )

    diag_payload = _run_capture_json(
        [
            sys.executable,
            "scripts/validation/run_predictive_hard_seed_diagnostics.py",
            "--scenario-matrix",
            str(scenario_matrix),
            "--seed-manifest",
            str(hard_seed_manifest),
            "--checkpoint",
            str(paths.checkpoint),
            "--horizon",
            str(int(eval_cfg.get("horizon", 120))),
            "--output-dir",
            str(paths.diagnostics_dir),
        ],
        log_level=args.log_level,
        allow_failure=True,
    )
    campaign_status: dict[str, Any] = {"status": "ok", "return_code": 0}
    try:
        _run(
            [
                sys.executable,
                "scripts/validation/run_predictive_success_campaign.py",
                "--checkpoints",
                str(paths.checkpoint),
                "--scenario-matrix",
                str(scenario_matrix),
                "--hard-seed-manifest",
                str(hard_seed_manifest),
                "--planner-grid",
                str(planner_grid),
                "--workers",
                str(int(eval_cfg.get("campaign_workers", 2))),
                "--horizon",
                str(int(eval_cfg.get("horizon", 120))),
                "--dt",
                str(float(eval_cfg.get("dt", 0.1))),
                "--output-dir",
                str(paths.campaign_dir),
            ],
            log_level=args.log_level,
        )
    except subprocess.CalledProcessError as exc:
        campaign_status = {
            "status": "failed",
            "return_code": int(exc.returncode),
            "command": [str(part) for part in exc.cmd] if exc.cmd else [],
        }

    campaign_summary_path = paths.campaign_dir / "campaign_summary.json"
    campaign_summary = {}
    if campaign_summary_path.exists():
        loaded_summary = json.loads(campaign_summary_path.read_text(encoding="utf-8"))
        if (
            int(campaign_status.get("return_code", 1)) == 0
            and loaded_summary.get("run_id") == run_id
            and str(loaded_summary.get("status", "")).lower() == "success"
        ):
            campaign_summary = loaded_summary

    stage_status = {
        "evaluation_ok": bool(int(eval_payload.get("return_code", 1)) == 0),
        "hard_seed_diagnostics_ok": bool(int(diag_payload.get("return_code", 1)) == 0),
        "campaign_ok": bool(int(campaign_status.get("return_code", 1)) == 0),
    }
    stage_status["all_ok"] = bool(
        stage_status["evaluation_ok"]
        and stage_status["hard_seed_diagnostics_ok"]
        and stage_status["campaign_ok"]
    )

    promoted_model: dict[str, Any] = {
        "attempted": False,
        "promoted": False,
        "model_id": str(train_cfg.get("model_id", "")),
        "checkpoint": str(paths.checkpoint),
    }
    if stage_status["all_ok"]:
        promoted_model["attempted"] = True
        register_payload = _run_capture_json(
            [
                sys.executable,
                "scripts/training/train_predictive_planner.py",
                "--dataset",
                str(paths.mixed_dataset),
                "--output-dir",
                str(paths.train_dir),
                "--model-id",
                str(train_cfg.get("model_id", f"predictive_proxy_selected_v2_{run_stamp}")),
                "--epochs",
                "0",
                "--checkpoint-only-register",
                str(paths.checkpoint),
                "--training-summary",
                str(training_summary_path),
                "--register-model",
            ],
            log_level=args.log_level,
            allow_failure=True,
        )
        promoted_model["register_payload"] = register_payload
        promoted_model["promoted"] = int(register_payload.get("return_code", 1)) == 0

    final_summary = {
        "contract_version": _CONTRACT_VERSION,
        "training_family": _TRAINING_FAMILY,
        "artifact_role": "predictive_pipeline_summary",
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "git_commit": git_commit,
        "scenario_matrix": str(scenario_matrix),
        "hard_seed_manifest": str(hard_seed_manifest),
        "planner_grid": str(planner_grid),
        "dataset_manifests": {
            "base": str(base_dataset_manifest),
            "hardcase": str(hardcase_dataset_manifest),
            "mixed": str(mixed_dataset_manifest),
        },
        "paths": {
            "run_root": str(paths.root),
            "base_dataset": str(paths.base_dataset),
            "hardcase_dataset": str(paths.hardcase_dataset),
            "mixed_dataset": str(paths.mixed_dataset),
            "train_dir": str(paths.train_dir),
            "checkpoint": str(paths.checkpoint),
            "eval_dir": str(paths.eval_dir),
            "diagnostics_dir": str(paths.diagnostics_dir),
            "campaign_dir": str(paths.campaign_dir),
        },
        "stage_status": stage_status,
        "training": training_summary,
        "evaluation": eval_payload,
        "hard_seed_diagnostics": diag_payload,
        "campaign": campaign_status,
        "success_campaign": campaign_summary,
        "promoted_model": promoted_model,
        "final_gate_results": {
            "all_ok": stage_status["all_ok"],
            "evaluation_ok": stage_status["evaluation_ok"],
            "hard_seed_diagnostics_ok": stage_status["hard_seed_diagnostics_ok"],
            "campaign_ok": stage_status["campaign_ok"],
        },
    }
    _write_json(paths.final_summary, final_summary)

    md_path = paths.root / "final_performance_summary.md"
    best = campaign_summary.get("best", {}) if isinstance(campaign_summary, dict) else {}
    best_hard = best.get("hard", {}) if isinstance(best, dict) else {}
    best_global = best.get("global", {}) if isinstance(best, dict) else {}
    lines = [
        "# Predictive Training Final Performance Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Contract version: `{_CONTRACT_VERSION}`",
        f"- Config: `{config_path}`",
        f"- Scenario matrix: `{scenario_matrix}`",
        f"- Checkpoint: `{paths.checkpoint}`",
        "",
        "## Stage Status",
        "",
        f"- evaluation_ok: `{stage_status['evaluation_ok']}`",
        f"- hard_seed_diagnostics_ok: `{stage_status['hard_seed_diagnostics_ok']}`",
        f"- campaign_ok: `{stage_status['campaign_ok']}`",
        f"- all_ok: `{stage_status['all_ok']}`",
        "",
        "## Final Campaign Snapshot",
        "",
        f"- Best variant: `{best.get('variant', '')}`",
        f"- Hard success: `{best_hard.get('success_rate', 'n/a')}`",
        f"- Global success: `{best_global.get('success_rate', 'n/a')}`",
        f"- Hard mean min-distance: `{best_hard.get('mean_min_distance', 'n/a')}`",
        f"- Global mean min-distance: `{best_global.get('mean_min_distance', 'n/a')}`",
        f"- Promoted model: `{promoted_model['promoted']}`",
        "",
        "## Artifacts",
        "",
        f"- JSON summary: `{paths.final_summary}`",
        f"- Base dataset manifest: `{base_dataset_manifest}`",
        f"- Hardcase dataset manifest: `{hardcase_dataset_manifest}`",
        f"- Mixed dataset manifest: `{mixed_dataset_manifest}`",
        f"- Eval dir: `{paths.eval_dir}`",
        f"- Diagnostics dir: `{paths.diagnostics_dir}`",
        f"- Campaign dir: `{paths.campaign_dir}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if stage_status["all_ok"]:
        logger.success("Pipeline complete. Final summary: {}", paths.final_summary)
    else:
        logger.warning(
            "Pipeline completed with failing stage gates. Summary: {}", paths.final_summary
        )
    print(
        json.dumps(
            {"run_root": str(paths.root), "final_summary": str(paths.final_summary)}, indent=2
        )
    )
    return 0 if stage_status["all_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
