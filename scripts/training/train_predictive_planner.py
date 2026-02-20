#!/usr/bin/env python3
"""Train the predictive planner trajectory model from collected rollout data."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

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
    parser.add_argument("--proxy-scenario-matrix", type=Path, default=None)
    parser.add_argument("--proxy-seed-manifest", type=Path, default=None)
    parser.add_argument("--proxy-every-epochs", type=int, default=0)
    parser.add_argument("--proxy-horizon", type=int, default=120)
    parser.add_argument("--proxy-dt", type=float, default=0.1)
    parser.add_argument("--proxy-workers", type=int, default=1)
    parser.add_argument("--select-by-proxy", action="store_true")
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

    for state_b, target_b, mask_b in loader:
        state_b = state_b.to(device)
        target_b = target_b.to(device)
        mask_b = mask_b.to(device)

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            out = model(state_b, mask_b)
            pred = out["future_positions"]
            loss = masked_trajectory_loss(pred, target_b, mask_b, horizon_weights=horizon_weights)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        ade, fde = compute_ade_fde(pred.detach(), target_b.detach(), mask_b.detach())
        losses.append(float(loss.item()))
        ade_vals.append(ade)
        fde_vals.append(fde)

    return float(np.mean(losses)), float(np.mean(ade_vals)), float(np.mean(fde_vals))


def main() -> int:  # noqa: C901, PLR0915
    """Train predictive trajectory model and persist checkpoint + metrics."""
    args = parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    raw = np.load(args.dataset)
    state = np.asarray(raw["state"], dtype=np.float32)
    target = np.asarray(raw["target"], dtype=np.float32)
    mask = np.asarray(raw["mask"], dtype=np.float32)

    if state.ndim != 3 or state.shape[-1] != 4:
        raise ValueError("Expected state shape (N, max_agents, 4)")
    if target.ndim != 4 or target.shape[-1] != 2:
        raise ValueError("Expected target shape (N, max_agents, horizon, 2)")

    train_loader, val_loader = _prepare_loaders(
        state=state,
        target=target,
        mask=mask,
        val_split=float(args.val_split),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
    )

    cfg = PredictiveModelConfig(
        max_agents=int(state.shape[1]),
        horizon_steps=int(target.shape[2]),
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
    history: list[dict[str, float]] = []
    proxy_history: list[dict[str, float | str]] = []
    best = {
        "val_loss": float("inf"),
        "val_ade": float("inf"),
        "val_fde": float("inf"),
        "epoch": -1,
    }
    best_proxy: dict[str, float] | None = None
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

        if val_loss < best["val_loss"]:
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
            if bool(args.select_by_proxy) and _proxy_better(proxy_metrics, best_proxy):
                best_proxy = {
                    "success_rate": float(proxy_metrics["success_rate"]),
                    "mean_min_distance": float(proxy_metrics["mean_min_distance"]),
                    "val_loss": float(val_loss),
                    "epoch": float(epoch),
                }
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

    summary = {
        "model_id": args.model_id,
        "dataset": str(args.dataset),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "config": asdict(cfg),
        "best": best,
        "quality_gates": gates,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "git_commit": _git_commit(),
        "selection_mode": "proxy" if bool(args.select_by_proxy) else "val_loss",
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
        "history": history,
    }

    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved training summary to {}", summary_path)

    if bool(args.register_model):
        rel_checkpoint = checkpoint_path
        try:
            rel_checkpoint = checkpoint_path.relative_to(Path.cwd())
        except ValueError:
            rel_checkpoint = checkpoint_path

        upsert_registry_entry(
            {
                "model_id": args.model_id,
                "display_name": f"Predictive planner model ({args.model_id})",
                "local_path": str(rel_checkpoint),
                "config_path": "",
                "commit": _git_commit(),
                "wandb_run_id": "",
                "wandb_run_path": "",
                "wandb_entity": "",
                "wandb_project": "",
                "wandb_file": "",
                "tags": ["predictive", "rgl", "planner", "trajectory"],
                "notes": [
                    f"Dataset: {args.dataset}",
                    "Generated by scripts/training/train_predictive_planner.py",
                ],
            }
        )
        logger.info("Updated model registry entry '{}'", args.model_id)

    if not gates["pass_all"]:
        logger.error("Quality gates failed: {}", gates)
        return 2
    logger.success("Quality gates passed: {}", gates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
