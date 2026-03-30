"""Summarize PPO num_envs benchmark runs from Weights & Biases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _coerce_float(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _extract_num_envs(run: Any) -> int | None:
    config = getattr(run, "config", {}) or {}
    return _coerce_int(config.get("num_envs"))


def _row_from_run(run: Any) -> dict[str, object]:
    summary = dict(getattr(run, "summary", {}) or {})
    return {
        "run_id": run.id,
        "name": run.name,
        "state": run.state,
        "num_envs": _extract_num_envs(run),
        "timesteps": _coerce_int(summary.get("time/total_timesteps")),
        "train_env_steps_per_sec": _coerce_float(summary.get("perf/train_env_steps_per_sec")),
        "fps": _coerce_float(summary.get("time/fps")),
        "success_rate": _coerce_float(summary.get("eval/success_rate")),
        "collision_rate": _coerce_float(summary.get("eval/collision_rate")),
        "snqi": _coerce_float(summary.get("eval/snqi")),
    }


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return "-"


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# PPO num_envs benchmark summary",
        "",
        "| num_envs | run_id | state | timesteps | env_steps/s | fps | success | collision | snqi |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {num_envs} | {run_id} | {state} | {timesteps} | {train_env_steps_per_sec} | {fps} | {success_rate} | {collision_rate} | {snqi} |".format(
                num_envs=row.get("num_envs", "-"),
                run_id=row["run_id"],
                state=row["state"],
                timesteps=_format_value(row["timesteps"]),
                train_env_steps_per_sec=_format_value(row["train_env_steps_per_sec"]),
                fps=_format_value(row["fps"]),
                success_rate=_format_value(row["success_rate"]),
                collision_rate=_format_value(row["collision_rate"]),
                snqi=_format_value(row["snqi"]),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Fetch grouped W&B runs and write compact JSON/Markdown benchmark summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    import wandb  # type: ignore

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})
    rows = sorted(
        (_row_from_run(run) for run in runs),
        key=lambda row: (-(row["num_envs"] or -1), str(row["run_id"])),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, rows)
    print(
        json.dumps(
            {"output_json": str(args.output_json), "output_md": str(args.output_md)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
