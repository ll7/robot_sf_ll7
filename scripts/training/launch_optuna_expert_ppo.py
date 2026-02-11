"""Launch Optuna expert PPO sweeps from a YAML launcher config."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

import yaml
from loguru import logger
from sqlalchemy.engine import make_url

_OBJECTIVE_MODES = ("best_checkpoint", "final_eval", "last_n_mean", "auc")
_CONSTRAINT_HANDLING_CHOICES = ("penalize", "prune")
_LOG_LEVEL_CHOICES = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")

_SUPPORTED_LAUNCH_KEYS = {
    "base_config",
    "trials",
    "metric",
    "objective_mode",
    "objective_window",
    "trial_timesteps",
    "eval_every",
    "eval_episodes",
    "study_name",
    "storage",
    "seed",
    "constraint_collision_rate_max",
    "constraint_comfort_exposure_max",
    "constraint_handling",
    "log_level",
    "disable_wandb",
    "deterministic",
}
_INT_KEYS = {
    "trials",
    "objective_window",
    "trial_timesteps",
    "eval_every",
    "eval_episodes",
    "seed",
}
_SCALAR_FLAG_MAP = {
    "trials": "--trials",
    "metric": "--metric",
    "objective_mode": "--objective-mode",
    "objective_window": "--objective-window",
    "trial_timesteps": "--trial-timesteps",
    "eval_every": "--eval-every",
    "eval_episodes": "--eval-episodes",
    "study_name": "--study-name",
    "storage": "--storage",
    "seed": "--seed",
    "constraint_collision_rate_max": "--constraint-collision-rate-max",
    "constraint_comfort_exposure_max": "--constraint-comfort-exposure-max",
    "constraint_handling": "--constraint-handling",
    "log_level": "--log-level",
}


def _resolve_optional_path(path: Path, raw: object, *, field_name: str) -> Path | None:
    """Resolve an optional path field relative to a launch config file."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    if path.parent == path:
        raise ValueError(f"Unable to resolve relative path for {field_name}")
    return (path.parent / candidate).resolve()


def _coerce_int(value: object, *, field_name: str) -> int:
    """Coerce a numeric launch value to an integer."""
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer for '{field_name}', received {value!r}") from exc


def _as_bool(value: object) -> bool:
    """Coerce booleans from YAML/CLI values with string-safe semantics."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
        raise ValueError(
            f"Invalid boolean string value {value!r}; expected one of true/false/1/0/yes/no/on/off."
        )
    return bool(value)


def _mask_storage_fallback(raw_value: str) -> str:
    """Best-effort redaction for malformed storage strings that fail URL parsing."""
    redacted = re.sub(r"(://)[^@/\s]+@", r"\1***@", raw_value)
    redacted = re.sub(r"(^|[\s])[^@/\s:]+:[^@/\s]+@", r"\1***@", redacted)
    return redacted


def _mask_storage_in_command(command: list[str]) -> str:
    """Render shell command while masking credentials in --storage URL values."""
    masked: list[str] = []
    i = 0
    while i < len(command):
        token = command[i]
        masked_token = token
        if token == "--storage" and i + 1 < len(command):
            storage_raw = command[i + 1]
            try:
                storage_url = make_url(storage_raw)
                masked.extend([token, storage_url.render_as_string(hide_password=True)])
            except Exception:
                masked.extend([token, _mask_storage_fallback(storage_raw)])
            i += 2
            continue
        if token.startswith("--storage="):
            storage_raw = token.split("=", maxsplit=1)[1]
            try:
                storage_url = make_url(storage_raw)
                masked_token = f"--storage={storage_url.render_as_string(hide_password=True)}"
            except Exception:
                masked_token = f"--storage={_mask_storage_fallback(storage_raw)}"
        masked.append(masked_token)
        i += 1
    return shlex.join(masked)


def load_launch_config(path: Path) -> dict[str, object]:
    """Load and validate launcher YAML payload."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Optuna launcher config must be a mapping.")
    payload = dict(raw)
    unknown = sorted(set(payload) - _SUPPORTED_LAUNCH_KEYS)
    if unknown:
        raise ValueError(f"Unknown launcher config key(s): {', '.join(unknown)}")
    if "base_config" not in payload:
        raise ValueError("Launcher config must define 'base_config'.")
    return payload


def _effective_value(payload: dict[str, object], args: argparse.Namespace, key: str) -> object:
    """Return override value when set, otherwise launcher-config value."""
    override = getattr(args, key)
    return payload.get(key) if override is None else override


def build_optuna_cli_args(
    *,
    launch_config_path: Path,
    payload: dict[str, object],
    args: argparse.Namespace,
) -> list[str]:
    """Translate launcher config and CLI overrides into Optuna runner arguments."""
    base_config = _resolve_optional_path(
        launch_config_path,
        _effective_value(payload, args, "base_config"),
        field_name="base_config",
    )
    if base_config is None:
        raise ValueError("base_config cannot be empty.")

    cli_args: list[str] = ["--config", str(base_config)]
    for key, flag in _SCALAR_FLAG_MAP.items():
        value = _effective_value(payload, args, key)
        if value is None:
            continue
        if key in _INT_KEYS:
            value = _coerce_int(value, field_name=key)
        cli_args.extend([flag, str(value)])

    disable_wandb = _effective_value(payload, args, "disable_wandb")
    if _as_bool(disable_wandb):
        cli_args.append("--disable-wandb")
    deterministic = _effective_value(payload, args, "deterministic")
    if _as_bool(deterministic):
        cli_args.append("--deterministic")

    return cli_args


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for launcher script."""
    parser = argparse.ArgumentParser(
        description="Launch scripts/training/optuna_expert_ppo.py from YAML settings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/ppo_imitation/optuna_expert_ppo.yaml"),
        help="Launcher YAML path (default: configs/training/ppo_imitation/optuna_expert_ppo.yaml).",
    )
    parser.add_argument(
        "--base-config", type=Path, default=None, help="Override base expert config."
    )
    parser.add_argument("--trials", type=int, default=None, help="Override trial count.")
    parser.add_argument("--metric", type=str, default=None, help="Override optimization metric.")
    parser.add_argument(
        "--objective-mode",
        choices=_OBJECTIVE_MODES,
        default=None,
        help="Override objective reducer mode.",
    )
    parser.add_argument(
        "--objective-window",
        type=int,
        default=None,
        help="Override last-N window for objective_mode=last_n_mean.",
    )
    parser.add_argument(
        "--trial-timesteps",
        type=int,
        default=None,
        help="Override timesteps per trial.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Override eval cadence in environment steps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override evaluation episodes per checkpoint.",
    )
    parser.add_argument("--study-name", type=str, default=None, help="Override Optuna study name.")
    parser.add_argument("--storage", type=str, default=None, help="Override Optuna storage URL.")
    parser.add_argument("--seed", type=int, default=None, help="Override Optuna sampler seed.")
    parser.add_argument(
        "--constraint-collision-rate-max",
        type=float,
        default=None,
        help="Optional safety gate: require collision_rate <= threshold.",
    )
    parser.add_argument(
        "--constraint-comfort-exposure-max",
        type=float,
        default=None,
        help="Optional safety gate: require comfort_exposure <= threshold.",
    )
    parser.add_argument(
        "--constraint-handling",
        type=str,
        choices=_CONSTRAINT_HANDLING_CHOICES,
        default=None,
        help="Infeasible trial strategy for safety constraints (penalize|prune).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=_LOG_LEVEL_CHOICES,
        default=None,
        help="Override Optuna console log level.",
    )
    parser.add_argument(
        "--disable-wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable W&B in trial runs.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable deterministic trial seeds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command without executing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the launcher."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    launch_config_path = args.config.resolve()
    payload = load_launch_config(launch_config_path)
    effective_log_level = str(_effective_value(payload, args, "log_level") or "WARNING").upper()
    logger.remove()
    logger.add(sys.stderr, level=effective_log_level)
    cli_args = build_optuna_cli_args(
        launch_config_path=launch_config_path,
        payload=payload,
        args=args,
    )

    optuna_script = Path(__file__).with_name("optuna_expert_ppo.py").resolve()
    command = [sys.executable, str(optuna_script), *cli_args]
    logger.info("Resolved Optuna command: {}", _mask_storage_in_command(command))
    if args.dry_run:
        return 0

    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
