"""Analyze PPO expert eval timeline artifacts for consistency and trend reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.imitation_manifest import get_training_run_manifest_path
from robot_sf.common.artifact_paths import get_artifact_root

_CORE_KEYS = (
    "success_rate",
    "collision_rate",
    "path_efficiency",
    "comfort_exposure",
    "snqi",
    "eval_episode_return",
    "eval_avg_step_reward",
)


def _resolve_path(path_value: str | None) -> Path:
    if not path_value:
        raise ValueError("Manifest does not include eval_timeline_path.")
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (get_artifact_root() / candidate).resolve()


def _load_manifest(run_id: str) -> dict[str, Any]:
    manifest_path = get_training_run_manifest_path(run_id)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found for run_id '{run_id}': {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training manifest must be a JSON object.")
    return payload


def _analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    steps = [int(row.get("eval_step", 0)) for row in rows]
    monotonic = steps == sorted(steps)
    unique = len(set(steps)) == len(steps)

    missing_by_key = dict.fromkeys(_CORE_KEYS, 0)
    for row in rows:
        for key in _CORE_KEYS:
            value = row.get(key)
            if value is None:
                missing_by_key[key] += 1

    return {
        "rows": len(rows),
        "monotonic_eval_step": monotonic,
        "unique_eval_step": unique,
        "missing_core_values": missing_by_key,
        "first_eval_step": (steps[0] if steps else None),
        "last_eval_step": (steps[-1] if steps else None),
    }


def _write_markdown(
    path: Path, *, run_id: str, timeline_path: Path, analysis: dict[str, Any]
) -> None:
    lines = [
        f"# PPO Eval Timeline Analysis: `{run_id}`",
        "",
        f"- source: `{timeline_path}`",
        f"- rows: `{analysis['rows']}`",
        f"- monotonic_eval_step: `{analysis['monotonic_eval_step']}`",
        f"- unique_eval_step: `{analysis['unique_eval_step']}`",
        f"- first_eval_step: `{analysis['first_eval_step']}`",
        f"- last_eval_step: `{analysis['last_eval_step']}`",
        "",
        "## Missing Core Values",
        "",
    ]
    for key, count in analysis["missing_core_values"].items():
        lines.append(f"- {key}: {count}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run CLI analysis for a training run's canonical eval timeline artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Training run ID from imitation manifest.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--output-md", type=Path, default=None, help="Optional output Markdown path."
    )
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.run_id)
    timeline_path = _resolve_path(manifest.get("eval_timeline_path"))
    rows_raw = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(rows_raw, list):
        raise ValueError("Eval timeline JSON must be a list of rows.")
    rows = [row for row in rows_raw if isinstance(row, dict)]
    analysis = _analyze_rows(rows)

    output_json = args.output_json or timeline_path.with_name(f"{args.run_id}.analysis.json")
    output_md = args.output_md or timeline_path.with_name(f"{args.run_id}.analysis.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "source": str(timeline_path),
                "analysis": analysis,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_markdown(output_md, run_id=args.run_id, timeline_path=timeline_path, analysis=analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
