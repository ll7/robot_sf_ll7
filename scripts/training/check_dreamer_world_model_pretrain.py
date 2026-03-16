"""Quality gate for Dreamer world-model pretraining runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _extract_world_model_losses(history: list[dict[str, Any]]) -> list[float]:
    """Collect finite world-model losses from the Dreamer run history."""
    values: list[float] = []
    for entry in history:
        observability = entry.get("observability")
        if not isinstance(observability, dict):
            continue
        raw = observability.get("learners/default_policy/WORLD_MODEL_L_total")
        if isinstance(raw, int | float) and math.isfinite(float(raw)):
            values.append(float(raw))
    return values


def _extract_throughputs(history: list[dict[str, Any]]) -> list[float]:
    """Collect finite trained-step throughput values."""
    values: list[float] = []
    for entry in history:
        observability = entry.get("observability")
        if not isinstance(observability, dict):
            continue
        raw = observability.get("learners/default_policy/num_module_steps_trained_throughput")
        if isinstance(raw, int | float) and math.isfinite(float(raw)):
            values.append(float(raw))
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Validate a Dreamer world-model pretrain run.")
    parser.add_argument("--run-summary", type=Path, required=True, help="Path to run_summary.json")
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=10,
        help="Minimum number of completed iterations required.",
    )
    parser.add_argument(
        "--max-final-loss-ratio",
        type=float,
        default=0.75,
        help="Require final world-model loss <= first loss * this ratio.",
    )
    parser.add_argument(
        "--min-loss-drop-abs",
        type=float,
        default=100.0,
        help="Require at least this absolute drop in world-model loss.",
    )
    parser.add_argument(
        "--min-throughput",
        type=float,
        default=1.0,
        help="Require median trained-step throughput above this threshold.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    payload = json.loads(args.run_summary.read_text(encoding="utf-8"))
    history = payload.get("history")
    if not isinstance(history, list):
        raise RuntimeError("run_summary.json does not contain a valid history array.")

    losses = _extract_world_model_losses(history)
    throughputs = _extract_throughputs(history)
    if len(history) < int(args.min_iterations):
        raise RuntimeError(
            f"Only {len(history)} iterations completed; expected at least {args.min_iterations}."
        )
    if len(losses) < 2:
        raise RuntimeError("Need at least two finite world-model loss points for gating.")

    first_loss = losses[0]
    final_loss = losses[-1]
    loss_ratio = final_loss / first_loss if first_loss != 0 else math.inf
    loss_drop = first_loss - final_loss
    median_throughput = sorted(throughputs)[len(throughputs) // 2] if throughputs else 0.0

    failures: list[str] = []
    if loss_ratio > float(args.max_final_loss_ratio):
        failures.append(
            f"final/initial world-model loss ratio {loss_ratio:.4f} exceeds "
            f"{float(args.max_final_loss_ratio):.4f}"
        )
    if loss_drop < float(args.min_loss_drop_abs):
        failures.append(
            f"absolute world-model loss drop {loss_drop:.4f} is below "
            f"{float(args.min_loss_drop_abs):.4f}"
        )
    if median_throughput < float(args.min_throughput):
        failures.append(
            f"median trained-step throughput {median_throughput:.4f} is below "
            f"{float(args.min_throughput):.4f}"
        )

    result = {
        "pass": not failures,
        "iterations_completed": len(history),
        "first_world_model_loss": first_loss,
        "final_world_model_loss": final_loss,
        "world_model_loss_ratio": loss_ratio,
        "world_model_loss_drop_abs": loss_drop,
        "median_trained_step_throughput": median_throughput,
        "last_checkpoint_path": payload.get("last_checkpoint_path"),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
