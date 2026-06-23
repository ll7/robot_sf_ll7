#!/usr/bin/env python3
"""Inventory and optionally smoke-test legacy PPO snapshots against Gymnasium.

The default mode is intentionally cheap: it verifies that legacy PPO checkpoints
that should remain supported are represented by durable registry entries, and it
records root-local debug snapshots as explicitly unsupported.  Pass
``--smoke-model-id`` for a hydrated/downloadable checkpoint smoke that loads the
model, predicts one action, and executes one current ``make_robot_env`` step.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from robot_sf.models.registry import get_registry_entry, load_registry, resolve_model_path

SUPPORTED_LEGACY_PPO_MODEL_IDS = (
    "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
    "ppo_expert_br06_v2_15m_all_maps_20260302T152332",
    "ppo_expert_br06_v2_15m_all_maps_20260303T074433",
)

UNSUPPORTED_ROOT_LOCAL_PPO_SNAPSHOTS = {
    "model/run_023.zip": "legacy debug checkpoint with no durable registry provenance",
    "model/run_043.zip": "legacy debug checkpoint with no durable registry provenance",
    "model/ppo_model_retrained_10m_2024-09-17.zip": (
        "root-local retrained checkpoint with no durable registry provenance"
    ),
    "model/ppo_model_retrained_10m_2025-02-01.zip": (
        "root-local retrained checkpoint with no durable registry provenance"
    ),
}


@dataclass(frozen=True)
class SnapshotRow:
    """One legacy snapshot support-status row."""

    identifier: str
    status: str
    source: str
    local_path: str
    durable_uri: str
    reason: str


@dataclass(frozen=True)
class SmokeReport:
    """One optional model-load and Gymnasium step smoke result."""

    model_id: str
    status: str
    model_path: str
    observation_space: str
    action_space: str
    action_shape: tuple[int, ...]
    reward_type: str
    terminated_type: str
    truncated_type: str
    info_keys: tuple[str, ...]


def _release_uri(entry: Mapping[str, Any]) -> str:
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return ""
    url = str(release.get("url") or "").strip()
    if url:
        return url
    repo = str(release.get("repo") or "").strip()
    tag = str(release.get("tag") or "").strip()
    asset_name = str(release.get("asset_name") or "").strip()
    if repo and tag and asset_name:
        return f"https://github.com/{repo}/releases/download/{tag}/{asset_name}"
    return ""


def _durable_release_reason(entry: Mapping[str, Any]) -> str:
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return "missing github_release pointer"
    missing = [
        field
        for field in ("asset_name", "sha256", "size_bytes")
        if str(release.get(field) or "").strip() == ""
    ]
    if missing:
        return f"github_release missing {', '.join(missing)}"
    return ""


def build_inventory(
    *,
    repo_root: Path,
    registry_path: Path,
    supported_model_ids: tuple[str, ...] = SUPPORTED_LEGACY_PPO_MODEL_IDS,
) -> tuple[SnapshotRow, ...]:
    """Return support-status rows for legacy PPO snapshots."""
    registry = load_registry(registry_path)
    rows: list[SnapshotRow] = []
    for model_id in supported_model_ids:
        entry = registry.get(model_id)
        if entry is None:
            rows.append(
                SnapshotRow(
                    identifier=model_id,
                    status="missing_registry_entry",
                    source="model_registry",
                    local_path="",
                    durable_uri="",
                    reason="supported legacy checkpoint is absent from model/registry.yaml",
                )
            )
            continue
        reason = _durable_release_reason(entry)
        rows.append(
            SnapshotRow(
                identifier=model_id,
                status="supported" if not reason else "unsupported_missing_durable_pointer",
                source="model_registry",
                local_path=str(entry.get("local_path") or ""),
                durable_uri=_release_uri(entry),
                reason=reason or "durable GitHub release pointer with checksum",
            )
        )

    for rel_path, reason in UNSUPPORTED_ROOT_LOCAL_PPO_SNAPSHOTS.items():
        rows.append(
            SnapshotRow(
                identifier=rel_path,
                status="unsupported_local_only",
                source="root_local_file"
                if (repo_root / rel_path).exists()
                else "root_local_missing",
                local_path=rel_path,
                durable_uri="",
                reason=reason,
            )
        )
    return tuple(rows)


def _load_ppo_model(model_path: Path):
    """Load a Stable-Baselines3 PPO checkpoint for the opt-in smoke path."""
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("stable_baselines3 is required for --smoke-model-id") from exc
    return PPO.load(str(model_path), env=None, device="cpu", print_system_info=False)


def _make_smoke_env(seed: int):
    """Create the current Gymnasium robot env for the opt-in smoke path."""
    from robot_sf.gym_env.environment_factory import make_robot_env
    from robot_sf.gym_env.unified_config import RobotSimulationConfig

    return make_robot_env(config=RobotSimulationConfig(map_id="uni_campus_big"), seed=seed)


def run_model_step_smoke(
    *,
    model_id: str,
    repo_root: Path,
    registry_path: Path,
    allow_download: bool,
    seed: int,
) -> SmokeReport:
    """Load a PPO checkpoint and execute one current Gymnasium robot-env step."""

    get_registry_entry(model_id, registry_path)
    model_path = resolve_model_path(
        model_id,
        registry_path=registry_path,
        allow_download=allow_download,
    )
    env = _make_smoke_env(seed)
    try:
        model = _load_ppo_model(model_path)
        obs, _reset_info = env.reset(seed=seed)
        raw_action, _state = model.predict(obs, deterministic=True)
        action = np.asarray(raw_action, dtype=getattr(env.action_space, "dtype", np.float32))
        if not env.action_space.contains(action):
            raise ValueError(
                f"Model action is outside current env action_space: action={action!r}, "
                f"space={env.action_space!r}"
            )
        step_obs, reward, terminated, truncated, info = env.step(action)
        if not env.observation_space.contains(step_obs):
            raise ValueError("Step observation is outside current env observation_space")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise TypeError("Gymnasium step must return bool terminated/truncated flags")
        if not isinstance(info, Mapping):
            raise TypeError("Gymnasium step info must be a mapping")
        return SmokeReport(
            model_id=model_id,
            status="ok",
            model_path=str(model_path),
            observation_space=type(env.observation_space).__name__,
            action_space=type(env.action_space).__name__,
            action_shape=tuple(action.shape),
            reward_type=type(reward).__name__,
            terminated_type=type(terminated).__name__,
            truncated_type=type(truncated).__name__,
            info_keys=tuple(sorted(str(key) for key in info.keys())),
        )
    finally:
        env.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--registry-path", type=Path, default=Path("model/registry.yaml"))
    parser.add_argument(
        "--smoke-model-id",
        action="append",
        default=[],
        help="Registry model id to load and step once. May be repeated.",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--seed", type=int, default=3469)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the inventory and optional smoke checks."""
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    registry_path = args.registry_path
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path

    inventory = build_inventory(repo_root=repo_root, registry_path=registry_path)
    smoke_reports = [
        run_model_step_smoke(
            model_id=model_id,
            repo_root=repo_root,
            registry_path=registry_path,
            allow_download=bool(args.allow_download),
            seed=args.seed,
        )
        for model_id in args.smoke_model_id
    ]
    blocking_rows = [
        row
        for row in inventory
        if row.identifier in SUPPORTED_LEGACY_PPO_MODEL_IDS and row.status != "supported"
    ]
    payload = {
        "schema": "legacy_ppo_snapshot_parity.v1",
        "status": "failed" if blocking_rows else "ok",
        "inventory": [asdict(row) for row in inventory],
        "smoke": [asdict(report) for report in smoke_reports],
        "blocking_rows": [asdict(row) for row in blocking_rows],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "legacy PPO snapshot parity: "
            f"status={payload['status']} supported={len(SUPPORTED_LEGACY_PPO_MODEL_IDS)} "
            f"smoke={len(smoke_reports)} blocking={len(blocking_rows)}"
        )
        for row in inventory:
            print(f"- {row.status}: {row.identifier} ({row.reason})")
    return 2 if blocking_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
