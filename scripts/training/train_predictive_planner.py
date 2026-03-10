#!/usr/bin/env python3
"""Train the predictive planner trajectory model from collected rollout data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from robot_sf.benchmark.map_runner import run_map_batch
from robot_sf.benchmark.predictive_planner_config import build_predictive_planner_algo_config
from robot_sf.models.registry import upsert_registry_entry
from robot_sf.planner.predictive_model import (
    PredictiveModelConfig,
    PredictiveTrajectoryModel,
    compute_ade_fde,
    masked_trajectory_loss,
    save_predictive_checkpoint,
)
from robot_sf.training.scenario_loader import load_scenarios

_CONTRACT_VERSION = "benchmark-reset-v2"
_TRAINING_FAMILY = "prediction_planner"


def _is_near_constant(arr: np.ndarray, *, tol: float = 1e-6) -> bool:
    """Return True when array spread is effectively zero."""
    if arr.size == 0:
        return True
    return float(np.nanmax(arr) - np.nanmin(arr)) <= float(tol)


def _dataset_diagnostics(
    *,
    state: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, Any]:
    """Compute dataset integrity diagnostics used for fail-fast validation."""
    flat_target = target.reshape(-1, target.shape[-1])
    valid_target = target_mask > 0.0
    disp = np.linalg.norm(target[:, :, -1, :] - target[:, :, 0, :], axis=-1)
    disp_valid = disp[target_mask[:, :, 0] > 0.0] if target_mask.ndim == 3 else disp.reshape(-1)

    diagnostics = {
        "num_samples": int(state.shape[0]),
        "max_agents": int(state.shape[1]),
        "horizon_steps": int(target.shape[2]),
        "active_agent_ratio": float(np.mean(mask)),
        "active_target_ratio": float(np.mean(target_mask)),
        "target_std": float(np.std(flat_target)),
        "target_near_constant": bool(_is_near_constant(flat_target)),
        "displacement_mean": float(np.mean(disp_valid)) if disp_valid.size > 0 else 0.0,
        "displacement_p95": float(np.percentile(disp_valid, 95)) if disp_valid.size > 0 else 0.0,
        "valid_target_entries": int(np.count_nonzero(valid_target)),
        "fail_reasons": [],
        "warnings": [],
    }

    fail_reasons: list[str] = []
    warnings: list[str] = []
    if diagnostics["num_samples"] < 64:
        fail_reasons.append("num_samples_below_64")
    if diagnostics["active_target_ratio"] <= 0.02:
        fail_reasons.append("active_target_ratio_too_low")
    if diagnostics["target_std"] <= 1e-6 or diagnostics["target_near_constant"]:
        fail_reasons.append("target_values_near_constant")
    if diagnostics["displacement_mean"] <= 1e-3:
        fail_reasons.append("trajectory_displacement_near_zero")
    if diagnostics["displacement_p95"] <= 1e-2:
        warnings.append("very_low_trajectory_spread")

    diagnostics["fail_reasons"] = fail_reasons
    diagnostics["warnings"] = warnings
    diagnostics["is_degenerate"] = bool(fail_reasons)
    return diagnostics


def parse_args() -> argparse.Namespace:
    """Parse CLI args for predictive model training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("output/tmp/predictive_planner/datasets/predictive_rollouts.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/tmp/predictive_planner/training/run_latest"),
    )
    parser.add_argument("--model-id", type=str, default="predictive_rgl_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--message-passing-steps", type=int, default=2)
    parser.add_argument("--max-val-ade", type=float, default=1.2)
    parser.add_argument("--max-val-fde", type=float, default=2.0)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument("--checkpoint-only-register", type=Path, default=None)
    parser.add_argument("--training-summary", type=Path, default=None)
    parser.add_argument("--proxy-scenario-matrix", type=Path, default=None)
    parser.add_argument("--proxy-seed-manifest", type=Path, default=None)
    parser.add_argument("--proxy-every-epochs", type=int, default=0)
    parser.add_argument("--proxy-horizon", type=int, default=120)
    parser.add_argument("--proxy-dt", type=float, default=0.1)
    parser.add_argument("--proxy-workers", type=int, default=1)
    parser.add_argument("--select-by-proxy", action="store_true")
    parser.add_argument("--wandb-enabled", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="robot_sf")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="predictive-training")
    parser.add_argument("--wandb-job-type", type=str, default="train")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-tags", nargs="*", default=())
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument(
        "--allow-degenerate-dataset",
        action="store_true",
        help="Allow training to continue even when dataset diagnostics detect degeneration.",
    )
    return parser.parse_args()


def _git_commit() -> str:
    """Resolve current git commit hash for provenance metadata."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def _episode_success(row: dict) -> bool:
    """Resolve episode success with collision-aware fallback semantics."""
    metrics = row.get("metrics", {})
    if "success_rate" in metrics:
        value = metrics.get("success_rate")
        if value is None or value == "":
            return False
        return float(value) >= 0.5
    value = metrics.get("success", False)
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return float(value) >= 0.5


def _load_seed_manifest(path: Path) -> dict[str, list[int]]:
    """Load scenario->seed map for hard-case proxy evaluation."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Seed manifest must be a mapping: {path}")
    out: dict[str, list[int]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            out[str(key)] = [int(v) for v in value]
    return out


def _make_subset_scenarios(
    scenario_matrix: Path, seed_manifest: dict[str, list[int]]
) -> list[dict]:
    """Load scenarios and apply explicit seed sets for proxy evaluation."""
    scenarios = load_scenarios(scenario_matrix)
    selected: list[dict] = []
    base_dir = scenario_matrix.parent.resolve()
    for scenario in scenarios:
        scenario_id = str(
            scenario.get("name") or scenario.get("scenario_id") or scenario.get("id") or "unknown"
        )
        if scenario_id not in seed_manifest:
            continue
        scenario_copy = dict(scenario)
        map_file = scenario_copy.get("map_file")
        if isinstance(map_file, str):
            map_path = Path(map_file)
            if not map_path.is_absolute():
                scenario_copy["map_file"] = str((base_dir / map_path).resolve())
        scenario_copy["seeds"] = list(seed_manifest[scenario_id])
        selected.append(scenario_copy)
    return selected


def _proxy_better(current: dict[str, float], best: dict[str, float] | None) -> bool:
    """Return True when current proxy metrics should replace best selection."""
    if best is None:
        return True
    if current["success_rate"] != best["success_rate"]:
        return current["success_rate"] > best["success_rate"]
    current_clearance = (
        current["mean_min_distance"] if np.isfinite(current["mean_min_distance"]) else float("-inf")
    )
    best_clearance = (
        best["mean_min_distance"] if np.isfinite(best["mean_min_distance"]) else float("-inf")
    )
    if current_clearance != best_clearance:
        return current_clearance > best_clearance
    return current["val_loss"] < best["val_loss"]


def _run_proxy_eval(
    *,
    checkpoint_path: Path,
    args: argparse.Namespace,
    epoch: int,
) -> dict[str, float]:
    """Run hard-case proxy benchmark on a checkpoint and return compact metrics."""
    if args.proxy_scenario_matrix is None:
        raise ValueError("Proxy scenario matrix is required for proxy evaluation.")
    scenarios_or_path: Path | list[dict]
    if args.proxy_seed_manifest is not None:
        seed_manifest = _load_seed_manifest(args.proxy_seed_manifest)
        scenarios_or_path = _make_subset_scenarios(args.proxy_scenario_matrix, seed_manifest)
        if not scenarios_or_path:
            raise RuntimeError(
                "Proxy seed manifest did not match any scenarios in "
                f"{args.proxy_scenario_matrix}: {args.proxy_seed_manifest}",
            )
    else:
        scenarios_or_path = args.proxy_scenario_matrix

    proxy_dir = args.output_dir / "proxy_eval"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    algo_cfg_path = proxy_dir / f"proxy_epoch_{epoch:04d}_algo.yaml"
    jsonl_path = proxy_dir / f"proxy_epoch_{epoch:04d}.jsonl"
    algo_cfg = build_predictive_planner_algo_config(
        checkpoint_path=checkpoint_path,
        device="cpu",
    )
    algo_cfg_path.write_text(yaml.safe_dump(algo_cfg, sort_keys=False), encoding="utf-8")
    if jsonl_path.exists():
        jsonl_path.unlink()

    run_map_batch(
        scenarios_or_path,
        jsonl_path,
        schema_path=Path("robot_sf/benchmark/schemas/episode.schema.v1.json"),
        algo="prediction_planner",
        algo_config_path=str(algo_cfg_path),
        horizon=int(args.proxy_horizon),
        dt=float(args.proxy_dt),
        workers=int(args.proxy_workers),
        resume=False,
        benchmark_profile="experimental",
    )
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("Proxy evaluation produced no episode rows.")
    success_vals = [1.0 if _episode_success(row) else 0.0 for row in rows]
    min_dist_vals = [
        float(row.get("metrics", {}).get("min_distance"))
        for row in rows
        if "min_distance" in row.get("metrics", {})
    ]
    return {
        "epoch": float(epoch),
        "episodes": float(len(rows)),
        "success_rate": float(np.mean(success_vals)),
        "mean_min_distance": float(np.mean(min_dist_vals)) if min_dist_vals else float("nan"),
        "jsonl_path": str(jsonl_path),
        "algo_config_path": str(algo_cfg_path),
    }


def _prepare_loaders(
    *,
    state: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    target_mask: np.ndarray,
    val_split: float,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders from numpy arrays."""
    n = state.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    val_n = int(max(1, round(n * val_split)))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]
    if train_idx.size == 0:
        train_idx = val_idx

    def _ds(sel: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(state[sel]).float(),
            torch.from_numpy(target[sel]).float(),
            torch.from_numpy(mask[sel]).float(),
            torch.from_numpy(target_mask[sel]).float(),
        )

    train_loader = DataLoader(_ds(train_idx), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(_ds(val_idx), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def _run_epoch(
    *,
    model: PredictiveTrajectoryModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run one train/eval epoch and return ``(loss, ade, fde)``."""
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    losses: list[float] = []
    ade_vals: list[float] = []
    fde_vals: list[float] = []

    horizon_weights = torch.linspace(1.5, 1.0, steps=model.config.horizon_steps, device=device)

    for state_b, target_b, mask_b, target_mask_b in loader:
        state_b = state_b.to(device)
        target_b = target_b.to(device)
        mask_b = mask_b.to(device)
        target_mask_b = target_mask_b.to(device)

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            out = model(state_b, mask_b)
            pred = out["future_positions"]
            loss = masked_trajectory_loss(
                pred,
                target_b,
                mask_b,
                target_mask_b,
                horizon_weights=horizon_weights,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        ade, fde = compute_ade_fde(
            pred.detach(),
            target_b.detach(),
            mask_b.detach(),
            target_mask_b.detach(),
        )
        losses.append(float(loss.item()))
        ade_vals.append(ade)
        fde_vals.append(fde)

    return float(np.mean(losses)), float(np.mean(ade_vals)), float(np.mean(fde_vals))


def _serialize_for_wandb(value: Any) -> Any:
    """Recursively convert dataclasses/paths into W&B-serializable primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize_for_wandb(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_wandb(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _init_wandb(args: argparse.Namespace, cfg: PredictiveModelConfig):
    """Initialize W&B run when enabled, otherwise return ``None``."""
    if not bool(args.wandb_enabled):
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency/runtime dependent
        raise RuntimeError(
            "W&B logging requested, but wandb is unavailable. Install extras and retry."
        ) from exc

    output_dir = Path(args.output_dir).resolve()
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run_name = str(args.wandb_name).strip() or str(args.model_id)
    entity = str(args.wandb_entity).strip() or None
    tags = [str(tag) for tag in args.wandb_tags if str(tag).strip()]
    config_payload = {
        "model_id": str(args.model_id),
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "val_split": float(args.val_split),
        "seed": int(args.seed),
        "select_by_proxy": bool(args.select_by_proxy),
        "proxy_scenario_matrix": str(args.proxy_scenario_matrix)
        if args.proxy_scenario_matrix
        else "",
        "proxy_seed_manifest": str(args.proxy_seed_manifest) if args.proxy_seed_manifest else "",
        "proxy_every_epochs": int(args.proxy_every_epochs),
        "model_config": asdict(cfg),
    }
    return wandb.init(
        project=str(args.wandb_project),
        entity=entity,
        group=str(args.wandb_group),
        job_type=str(args.wandb_job_type),
        name=run_name,
        mode=str(args.wandb_mode),
        dir=str(wandb_dir),
        tags=tags,
        config=_serialize_for_wandb(config_payload),
    )


def _selection_decision(
    *,
    select_by_proxy: bool,
    best: dict[str, float],
    best_proxy: dict[str, float] | None,
) -> dict[str, Any]:
    """Build deterministic checkpoint-selection metadata."""
    if bool(select_by_proxy) and best_proxy is not None:
        return {
            "selection_mode": "proxy",
            "selected_checkpoint_reason": (
                "Selected by proxy campaign ranking: success_rate, then mean_min_distance, "
                "then validation loss."
            ),
            "selected_epoch": int(best["epoch"]),
            "proxy_metrics": dict(best_proxy),
            "validation_metrics": {
                "val_loss": float(best["val_loss"]),
                "val_ade": float(best["val_ade"]),
                "val_fde": float(best["val_fde"]),
            },
        }
    return {
        "selection_mode": "val_loss",
        "selected_checkpoint_reason": "Selected by minimum validation loss.",
        "selected_epoch": int(best["epoch"]),
        "proxy_metrics": None,
        "validation_metrics": {
            "val_loss": float(best["val_loss"]),
            "val_ade": float(best["val_ade"]),
            "val_fde": float(best["val_fde"]),
        },
    }


def _should_update_val_loss_best(
    *,
    select_by_proxy: bool,
    proxy_selected: bool,
    val_loss: float,
    best_val_loss: float,
) -> bool:
    """Return whether the val-loss path should replace the current best checkpoint."""
    if bool(select_by_proxy) and bool(proxy_selected):
        return False
    return float(val_loss) < float(best_val_loss)


def _register_model_entry(
    *,
    model_id: str,
    checkpoint_path: Path,
    dataset: Path,
    summary_path: Path,
    selection: dict[str, Any],
) -> None:
    """Upsert registry metadata for a promoted predictive checkpoint."""
    rel_checkpoint = checkpoint_path
    try:
        rel_checkpoint = checkpoint_path.relative_to(Path.cwd())
    except ValueError:
        rel_checkpoint = checkpoint_path

    upsert_registry_entry(
        {
            "model_id": model_id,
            "display_name": f"Predictive planner model ({model_id})",
            "local_path": str(rel_checkpoint),
            "config_path": "",
            "commit": _git_commit(),
            "wandb_run_id": "",
            "wandb_run_path": "",
            "wandb_entity": "",
            "wandb_project": "",
            "wandb_file": "",
            "tags": ["predictive", "rgl", "planner", "trajectory", "benchmark-reset-v2"],
            "notes": [
                f"Dataset: {dataset}",
                f"Training summary: {summary_path}",
                f"Selection mode: {selection.get('selection_mode', 'unknown')}",
                "Promoted by scripts/training/train_predictive_planner.py",
            ],
        }
    )


def _dataset_manifest_path(dataset_path: Path) -> Path:
    """Return the expected dataset-manifest path for a generated NPZ dataset."""
    return dataset_path.with_suffix(dataset_path.suffix + ".manifest.json")


def _source_dataset_ids(dataset_path: Path) -> list[str]:
    """Resolve dataset provenance ids from the sibling manifest when available."""
    manifest_path = _dataset_manifest_path(dataset_path)
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to parse dataset manifest {}: {}", manifest_path, exc)
        else:
            if not isinstance(payload, dict):
                logger.warning(
                    "Dataset manifest {} must be a JSON object, got {}",
                    manifest_path,
                    type(payload).__name__,
                )
                return [f"{_TRAINING_FAMILY}:{dataset_path.stem}"]
            dataset_id = str(payload.get("dataset_id", "")).strip()
            if dataset_id:
                return [f"{_TRAINING_FAMILY}:{dataset_id}"]
            logger.warning("Dataset manifest {} is missing dataset_id", manifest_path)
    else:
        logger.warning("Dataset manifest missing next to dataset: {}", manifest_path)
    return [f"{_TRAINING_FAMILY}:{dataset_path.stem}"]


def _summary_path_candidates(summary: dict[str, object], *keys: str) -> list[Path]:
    """Return normalized path candidates from top-level summary keys and nested selection fields."""
    candidates: list[Path] = []
    selection = summary.get("selection", {})
    nested = selection if isinstance(selection, dict) else {}
    for key in keys:
        for value in (summary.get(key), nested.get(key)):
            text = str(value or "").strip()
            if text:
                candidates.append(Path(text).resolve())
    return candidates


def _validate_checkpoint_registration_inputs(
    *,
    summary: dict[str, object],
    checkpoint_path: Path,
    dataset_path: Path,
    model_id: str,
) -> None:
    """Ensure checkpoint-only registration matches the training summary provenance."""
    summary_checkpoints = _summary_path_candidates(summary, "checkpoint", "checkpoint_path")
    if not summary_checkpoints:
        raise RuntimeError("Training summary missing checkpoint provenance for registration.")
    if checkpoint_path.resolve() not in summary_checkpoints:
        raise RuntimeError(
            "Checkpoint does not match training summary provenance: "
            f"{checkpoint_path} not in {summary_checkpoints}"
        )

    summary_datasets = _summary_path_candidates(summary, "dataset")
    if not summary_datasets:
        raise RuntimeError("Training summary missing dataset provenance for registration.")
    if dataset_path.resolve() not in summary_datasets:
        raise RuntimeError(
            "Dataset does not match training summary provenance: "
            f"{dataset_path} not in {summary_datasets}"
        )

    selection = summary.get("selection", {})
    selection_dict = selection if isinstance(selection, dict) else {}
    summary_model_id = str(summary.get("model_id") or selection_dict.get("model_id") or "").strip()
    if not summary_model_id:
        raise RuntimeError("Training summary missing model_id provenance for registration.")
    if summary_model_id != model_id:
        raise RuntimeError(
            f"Model id does not match training summary provenance: {model_id} != {summary_model_id}"
        )


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    """Train predictive trajectory model and persist checkpoint + metrics."""
    args = parse_args()

    if args.checkpoint_only_register is not None:
        if args.training_summary is None:
            raise ValueError("--training-summary is required with --checkpoint-only-register")
        if not bool(args.register_model):
            raise ValueError("--register-model is required with --checkpoint-only-register")
        if not args.checkpoint_only_register.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_only_register}")
        if not args.training_summary.exists():
            raise FileNotFoundError(f"Training summary not found: {args.training_summary}")
        summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise TypeError(f"Expected JSON object at {args.training_summary}")
        _validate_checkpoint_registration_inputs(
            summary=summary,
            checkpoint_path=args.checkpoint_only_register,
            dataset_path=args.dataset,
            model_id=str(args.model_id),
        )
        gates = summary.get("quality_gates", {})
        if not isinstance(gates, dict) or not bool(gates.get("pass_all", False)):
            raise RuntimeError("Refusing to register predictive model with failing training gates.")
        selection = summary.get("selection", {})
        if not isinstance(selection, dict):
            selection = {}
        _register_model_entry(
            model_id=str(args.model_id),
            checkpoint_path=args.checkpoint_only_register,
            dataset=args.dataset,
            summary_path=args.training_summary,
            selection=selection,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "return_code": 0,
                    "model_id": str(args.model_id),
                    "checkpoint": str(args.checkpoint_only_register),
                }
            )
        )
        return 0

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw = np.load(args.dataset)
    state = np.asarray(raw["state"], dtype=np.float32)
    target = np.asarray(raw["target"], dtype=np.float32)
    mask = np.asarray(raw["mask"], dtype=np.float32)
    target_mask = (
        np.asarray(raw["target_mask"], dtype=np.float32)
        if "target_mask" in raw
        else np.repeat(mask[:, :, None], target.shape[2], axis=2).astype(np.float32)
    )

    if state.ndim != 3 or state.shape[-1] < 4:
        raise ValueError("Expected state shape (N, max_agents, F) with F >= 4")
    if target.ndim != 4 or target.shape[-1] != 2:
        raise ValueError("Expected target shape (N, max_agents, horizon, 2)")

    diagnostics = _dataset_diagnostics(
        state=state,
        target=target,
        mask=mask,
        target_mask=target_mask,
    )
    diagnostics_path = args.output_dir / "dataset_diagnostics.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    if diagnostics["warnings"]:
        logger.warning("Dataset diagnostics warnings: {}", diagnostics["warnings"])
    if diagnostics["is_degenerate"] and not bool(args.allow_degenerate_dataset):
        raise RuntimeError(
            "Dataset diagnostics flagged degeneracy: "
            f"{diagnostics['fail_reasons']}. "
            "Pass --allow-degenerate-dataset to bypass for debugging only."
        )

    train_loader, val_loader = _prepare_loaders(
        state=state,
        target=target,
        mask=mask,
        target_mask=target_mask,
        val_split=float(args.val_split),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    cfg = PredictiveModelConfig(
        max_agents=int(state.shape[1]),
        horizon_steps=int(target.shape[2]),
        input_dim=int(state.shape[2]),
        hidden_dim=int(args.hidden_dim),
        message_passing_steps=int(args.message_passing_steps),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PredictiveTrajectoryModel(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "predictive_model.pt"
    wandb_run = _init_wandb(args, cfg)
    history: list[dict[str, float]] = []
    proxy_history: list[dict[str, float | str]] = []
    best = {
        "val_loss": float("inf"),
        "val_ade": float("inf"),
        "val_fde": float("inf"),
        "epoch": -1,
    }
    best_proxy: dict[str, float] | None = None
    proxy_selected = False
    proxy_enabled = args.proxy_scenario_matrix is not None and int(args.proxy_every_epochs) > 0

    for epoch in range(1, int(args.epochs) + 1):
        train_loss, train_ade, train_fde = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_ade, val_fde = _run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_ade": train_ade,
            "train_fde": train_fde,
            "val_loss": val_loss,
            "val_ade": val_ade,
            "val_fde": val_fde,
        }
        history.append(row)

        if _should_update_val_loss_best(
            select_by_proxy=bool(args.select_by_proxy),
            proxy_selected=proxy_selected,
            val_loss=val_loss,
            best_val_loss=float(best["val_loss"]),
        ):
            best = {
                "val_loss": val_loss,
                "val_ade": val_ade,
                "val_fde": val_fde,
                "epoch": int(epoch),
            }
            save_predictive_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "val_loss": val_loss,
                    "val_ade": val_ade,
                    "val_fde": val_fde,
                },
                extra={
                    "dataset": str(args.dataset),
                    "model_id": args.model_id,
                    "selection_mode": "val_loss_fallback"
                    if bool(args.select_by_proxy)
                    else "val_loss",
                },
            )
        logger.info(
            "epoch={} train_loss={:.4f} val_loss={:.4f} val_ade={:.3f} val_fde={:.3f}",
            epoch,
            train_loss,
            val_loss,
            val_ade,
            val_fde,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": int(epoch),
                    "train/loss": float(train_loss),
                    "train/ade": float(train_ade),
                    "train/fde": float(train_fde),
                    "val/loss": float(val_loss),
                    "val/ade": float(val_ade),
                    "val/fde": float(val_fde),
                    "best/val_loss": float(best["val_loss"]),
                    "best/val_ade": float(best["val_ade"]),
                    "best/val_fde": float(best["val_fde"]),
                },
                step=int(epoch),
            )

        if proxy_enabled and epoch % int(args.proxy_every_epochs) == 0:
            epoch_ckpt_dir = args.output_dir / "proxy_checkpoints"
            epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
            epoch_ckpt = epoch_ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_predictive_checkpoint(
                epoch_ckpt,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "val_loss": val_loss,
                    "val_ade": val_ade,
                    "val_fde": val_fde,
                },
                extra={
                    "dataset": str(args.dataset),
                    "model_id": args.model_id,
                },
            )
            try:
                proxy_metrics = _run_proxy_eval(checkpoint_path=epoch_ckpt, args=args, epoch=epoch)
            except Exception:
                logger.exception(
                    "Proxy evaluation failed at epoch={} checkpoint={}",
                    epoch,
                    epoch_ckpt,
                )
                continue
            proxy_metrics["val_loss"] = float(val_loss)
            proxy_metrics["val_ade"] = float(val_ade)
            proxy_metrics["val_fde"] = float(val_fde)
            proxy_history.append(proxy_metrics)
            logger.info(
                "proxy epoch={} success_rate={:.3f} mean_min_distance={:.3f}",
                epoch,
                proxy_metrics["success_rate"],
                proxy_metrics["mean_min_distance"],
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": int(epoch),
                        "proxy/success_rate": float(proxy_metrics["success_rate"]),
                        "proxy/mean_min_distance": float(proxy_metrics["mean_min_distance"]),
                        "proxy/episodes": float(proxy_metrics["episodes"]),
                    },
                    step=int(epoch),
                )
            if bool(args.select_by_proxy) and _proxy_better(proxy_metrics, best_proxy):
                best_proxy = {
                    "success_rate": float(proxy_metrics["success_rate"]),
                    "mean_min_distance": float(proxy_metrics["mean_min_distance"]),
                    "val_loss": float(val_loss),
                    "epoch": float(epoch),
                }
                proxy_selected = True
                best = {
                    "val_loss": val_loss,
                    "val_ade": val_ade,
                    "val_fde": val_fde,
                    "epoch": int(epoch),
                }
                save_predictive_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics={
                        "val_loss": val_loss,
                        "val_ade": val_ade,
                        "val_fde": val_fde,
                        "proxy_success_rate": float(proxy_metrics["success_rate"]),
                        "proxy_mean_min_distance": float(proxy_metrics["mean_min_distance"]),
                    },
                    extra={
                        "dataset": str(args.dataset),
                        "model_id": args.model_id,
                        "selection_mode": "proxy",
                    },
                )

    if not checkpoint_path.exists():
        save_predictive_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=int(best["epoch"]) if best["epoch"] >= 0 else int(args.epochs),
            metrics={
                "val_loss": float(best["val_loss"]),
                "val_ade": float(best["val_ade"]),
                "val_fde": float(best["val_fde"]),
            },
            extra={
                "dataset": str(args.dataset),
                "model_id": args.model_id,
                "selection_mode": "post_train_fallback",
            },
        )

    gates = {
        "max_val_ade": float(args.max_val_ade),
        "max_val_fde": float(args.max_val_fde),
        "pass_val_ade": bool(best["val_ade"] <= float(args.max_val_ade)),
        "pass_val_fde": bool(best["val_fde"] <= float(args.max_val_fde)),
    }
    gates["pass_all"] = bool(gates["pass_val_ade"] and gates["pass_val_fde"])
    selection = _selection_decision(
        select_by_proxy=bool(args.select_by_proxy),
        best=best,
        best_proxy=best_proxy,
    )

    summary = {
        "contract_version": _CONTRACT_VERSION,
        "training_family": _TRAINING_FAMILY,
        "artifact_role": "predictive_training_summary",
        "generated_at": datetime.now(UTC).isoformat(),
        "model_id": args.model_id,
        "dataset": str(args.dataset),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "config": asdict(cfg),
        "best": best,
        "best_checkpoint": {
            "path": str(checkpoint_path),
            "epoch": int(best["epoch"]),
        },
        "quality_gates": gates,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "git_commit": _git_commit(),
        "selection_mode": selection["selection_mode"],
        "selection": selection,
        "selected_checkpoint_reason": selection["selected_checkpoint_reason"],
        "source_dataset_ids": _source_dataset_ids(Path(args.dataset)),
        "proxy": {
            "enabled": bool(proxy_enabled),
            "scenario_matrix": str(args.proxy_scenario_matrix)
            if args.proxy_scenario_matrix
            else "",
            "seed_manifest": str(args.proxy_seed_manifest) if args.proxy_seed_manifest else "",
            "every_epochs": int(args.proxy_every_epochs),
            "horizon": int(args.proxy_horizon),
            "dt": float(args.proxy_dt),
            "workers": int(args.proxy_workers),
            "history": proxy_history,
            "best_proxy": best_proxy,
        },
        "dataset_stats": {
            "num_samples": int(state.shape[0]),
            "active_agent_ratio": float(np.mean(mask)),
            "active_target_ratio": float(np.mean(target_mask)),
        },
        "dataset_diagnostics_path": str(diagnostics_path),
        "dataset_diagnostics": diagnostics,
        "history": history,
    }

    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved training summary to {}", summary_path)
    if wandb_run is not None:
        wandb_run.summary["training_summary_path"] = str(summary_path)
        wandb_run.summary["dataset_diagnostics_path"] = str(diagnostics_path)
        wandb_run.summary["dataset_is_degenerate"] = bool(diagnostics["is_degenerate"])
        wandb_run.summary["checkpoint_path"] = str(checkpoint_path)
        wandb_run.summary["best_epoch"] = int(best["epoch"])
        wandb_run.summary["best_val_loss"] = float(best["val_loss"])
        wandb_run.summary["best_val_ade"] = float(best["val_ade"])
        wandb_run.summary["best_val_fde"] = float(best["val_fde"])
        wandb_run.summary["quality_gate_pass"] = bool(gates["pass_all"])

    if not gates["pass_all"]:
        logger.error("Quality gates failed: {}", gates)
        if wandb_run is not None:
            wandb_run.finish(exit_code=2)
        return 2
    logger.success("Quality gates passed: {}", gates)
    if bool(args.register_model):
        _register_model_entry(
            model_id=str(args.model_id),
            checkpoint_path=checkpoint_path,
            dataset=args.dataset,
            summary_path=summary_path,
            selection=selection,
        )
        logger.info("Updated model registry entry '{}'", args.model_id)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
