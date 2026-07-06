"""Summarize behavior-token distributions and surface representative example windows.

Joins the ``windows.jsonl`` (from ``extract_windows.py``) with token assignments
(from ``quantize_trace_windows.py``) and writes diagnostic artifacts: a per-token
summary broken down by scenario / planner / outcome / row status, a compact Markdown
description of frequent tokens with heuristic *candidate* motif labels, and bounded
example windows per frequent token.

Example::

    uv run python experiments/behavior_tokens/inspect_token_motifs.py \\
        --windows-jsonl output/experiments/behavior_tokens/windows.jsonl \\
        --assignments-csv output/experiments/behavior_tokens/token_assignments.csv \\
        --output-dir output/experiments/behavior_tokens/inspection

Claim boundary: motif labels are manual, heuristic *candidates* for inspection only.
Token ids are not ground-truth labels and carry no benchmark or safety authority.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.behavior_tokens.schemas import (  # noqa: E402
    CLAIM_BOUNDARY,
    FEATURE_NAMES,
    INSPECTION_SCHEMA_VERSION,
    NEAR_PEDESTRIAN_THRESHOLD_M,
    STOP_SPEED_THRESHOLD_M_S,
)


def load_windows(path: Path) -> dict[str, dict[str, Any]]:
    """Load windows JSONL keyed by ``window_id``."""
    windows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("window_id"):
                windows[str(record["window_id"])] = record
    return windows


def load_assignments(csv_path: Path | None, json_path: Path | None) -> dict[str, int]:
    """Load ``window_id -> token_id`` from a CSV or JSON assignments file."""
    if csv_path is not None:
        assignments: dict[str, int] = {}
        with open(csv_path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("window_id") and row.get("token_id") not in (None, ""):
                    assignments[str(row["window_id"])] = int(float(row["token_id"]))
        return assignments
    if json_path is not None:
        with open(json_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        items = payload.get("assignments", payload) if isinstance(payload, dict) else payload
        return {
            str(item["window_id"]): int(item["token_id"])
            for item in items
            if isinstance(item, dict) and "window_id" in item and "token_id" in item
        }
    raise ValueError("either --assignments-csv or --assignments-json must be provided")


def _mean_features(windows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Return the mean of each feature over windows, ignoring null values."""
    means: dict[str, float | None] = {}
    for name in FEATURE_NAMES:
        values = [
            float(w["features"][name])
            for w in windows
            if isinstance(w.get("features"), dict) and _is_number(w["features"].get(name))
        ]
        means[name] = float(sum(values) / len(values)) if values else None
    return means


def _is_number(value: Any) -> bool:
    """Return True when ``value`` is a real number (not bool/None)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def candidate_motif_label(mean_features: dict[str, float | None]) -> str:
    """Return a heuristic *candidate* motif label from mean feature values.

    Labels are deliberately suffixed ``_candidate`` because token ids are diagnostic,
    not ground-truth interaction classes. The ordering encodes a rough priority:
    the most safety-relevant motif that matches wins.
    """

    def val(name: str) -> float | None:
        return mean_features.get(name)

    clearance_min = val("clearance_min_m")
    stop_yield = val("stop_yield_fraction")
    oscillation = val("oscillation_rate")
    speed_mean = val("robot_speed_mean_m_s")
    speed_min = val("robot_speed_min_m_s")
    recovery = val("near_conflict_recovery_m")
    ttc_min = val("ttc_proxy_min_s")

    # Deadlock/freeze: sustained near-zero speed while close to a pedestrian.
    if (
        speed_mean is not None
        and speed_mean < STOP_SPEED_THRESHOLD_M_S
        and clearance_min is not None
        and clearance_min < NEAR_PEDESTRIAN_THRESHOLD_M
    ):
        return "deadlock_or_freeze_candidate"
    # Yielding: frequently stopping/slowing while a pedestrian is near.
    if stop_yield is not None and stop_yield >= 0.4:
        return "yielding_candidate"
    # Emergency-braking: very low clearance / TTC with a sharp slow-down.
    if (clearance_min is not None and clearance_min < 0.5) or (
        ttc_min is not None and ttc_min < 1.0
    ):
        return "emergency_braking_or_unsafe_cut_in_candidate"
    # Oscillatory negotiation: frequent steering sign changes.
    if oscillation is not None and oscillation >= 0.4:
        return "oscillatory_negotiation_candidate"
    # Assertive passing: maintaining speed with recovery after the closest point.
    if speed_mean is not None and speed_mean >= 1.0 and recovery is not None and recovery > 0.5:
        return "assertive_passing_candidate"
    # Low-speed maneuvering near a pedestrian without a full stop.
    if speed_min is not None and speed_min < STOP_SPEED_THRESHOLD_M_S:
        return "slow_maneuver_candidate"
    return "unlabeled_candidate"


def build_summary(
    windows: dict[str, dict[str, Any]], assignments: dict[str, int]
) -> dict[str, Any]:
    """Build the per-token summary structure joining windows and assignments."""
    by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    joined = 0
    for window_id, token_id in assignments.items():
        window = windows.get(window_id)
        if window is None:
            continue
        by_token[token_id].append(window)
        joined += 1

    tokens: list[dict[str, Any]] = []
    for token_id in sorted(by_token):
        members = by_token[token_id]
        mean_features = _mean_features(members)
        tokens.append(
            {
                "token_id": token_id,
                "count": len(members),
                "candidate_motif_label": candidate_motif_label(mean_features),
                "by_scenario": dict(Counter(str(w.get("scenario_id")) for w in members)),
                "by_planner": dict(Counter(str(w.get("planner_key")) for w in members)),
                "by_outcome": dict(Counter(str(w.get("outcome")) for w in members)),
                "by_row_status": dict(Counter(str(w.get("row_status")) for w in members)),
                "mean_features": mean_features,
            }
        )
    return {
        "inspection_schema_version": INSPECTION_SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "windows_total": len(windows),
        "assignments_total": len(assignments),
        "windows_joined": joined,
        "num_tokens": len(tokens),
        "tokens": tokens,
    }


def select_examples(
    windows: dict[str, dict[str, Any]],
    assignments: dict[str, int],
    token_id: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` representative windows for a token (stable order)."""
    members = [
        windows[wid]
        for wid, tid in sorted(assignments.items())
        if tid == token_id and wid in windows
    ]
    return members[:limit]


def _render_frequent_tokens_md(summary: dict[str, Any], min_count: int) -> str:
    """Render a compact Markdown description of frequent tokens."""
    lines = [
        "# Frequent behavior-token motifs (diagnostic candidates)",
        "",
        f"> {CLAIM_BOUNDARY}",
        "",
        f"Tokens with count >= {min_count} are shown. Motif labels are heuristic "
        "*candidates*, not validated interaction classes.",
        "",
    ]
    frequent = [t for t in summary["tokens"] if t["count"] >= min_count]
    if not frequent:
        lines.append("_No token met the frequency threshold._")
        return "\n".join(lines) + "\n"
    for token in sorted(frequent, key=lambda t: t["count"], reverse=True):
        lines.append(f"## Token {token['token_id']} — {token['candidate_motif_label']}")
        lines.append("")
        lines.append(f"- Count: {token['count']}")
        top_scen = sorted(token["by_scenario"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_outcome = sorted(token["by_outcome"].items(), key=lambda kv: kv[1], reverse=True)[:3]
        lines.append(f"- Top scenarios: {', '.join(f'{k} ({v})' for k, v in top_scen)}")
        lines.append(f"- Top outcomes: {', '.join(f'{k} ({v})' for k, v in top_outcome)}")
        key_feats = [
            "clearance_min_m",
            "robot_speed_mean_m_s",
            "stop_yield_fraction",
            "oscillation_rate",
            "ttc_proxy_min_s",
        ]
        feat_bits = []
        for name in key_feats:
            value = token["mean_features"].get(name)
            feat_bits.append(f"{name}={value:.3f}" if _is_number(value) else f"{name}=null")
        lines.append(f"- Mean features: {', '.join(feat_bits)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_inspection_outputs(
    output_dir: Path,
    windows: dict[str, dict[str, Any]],
    assignments: dict[str, int],
    summary: dict[str, Any],
    *,
    min_token_count: int,
    examples_per_token: int,
) -> dict[str, Any]:
    """Write summary, motif Markdown, and per-token example files. Returns a manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = output_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "token_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    with open(output_dir / "token_summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "token_id",
                "count",
                "candidate_motif_label",
                "n_scenarios",
                "n_planners",
                "n_outcomes",
            ]
        )
        for token in summary["tokens"]:
            writer.writerow(
                [
                    token["token_id"],
                    token["count"],
                    token["candidate_motif_label"],
                    len(token["by_scenario"]),
                    len(token["by_planner"]),
                    len(token["by_outcome"]),
                ]
            )

    with open(output_dir / "frequent_tokens.md", "w", encoding="utf-8") as handle:
        handle.write(_render_frequent_tokens_md(summary, min_token_count))

    example_files: list[str] = []
    for token in summary["tokens"]:
        if token["count"] < min_token_count:
            continue
        examples = select_examples(windows, assignments, token["token_id"], examples_per_token)
        example_path = examples_dir / f"token_{token['token_id']}_examples.jsonl"
        with open(example_path, "w", encoding="utf-8") as handle:
            for window in examples:
                handle.write(json.dumps(window, sort_keys=True))
                handle.write("\n")
        example_files.append(str(example_path))

    return {
        "token_summary_json": str(output_dir / "token_summary.json"),
        "token_summary_csv": str(output_dir / "token_summary.csv"),
        "frequent_tokens_md": str(output_dir / "frequent_tokens.md"),
        "example_files": example_files,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for motif inspection."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize behavior-token distributions and export representative example "
            "windows for manual inspection. Offline and read-only."
        ),
        epilog=CLAIM_BOUNDARY,
    )
    parser.add_argument(
        "--windows-jsonl",
        type=Path,
        required=True,
        help="Input windows JSONL from extract_windows.py.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--assignments-csv",
        type=Path,
        default=None,
        help="Token assignments CSV (window_id, token_id).",
    )
    group.add_argument(
        "--assignments-json",
        type=Path,
        default=None,
        help="Token assignments JSON (metadata + assignments).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments/behavior_tokens/inspection"),
        help="Directory for inspection artifacts.",
    )
    parser.add_argument(
        "--min-token-count",
        type=int,
        default=5,
        help="Minimum count for a token to be 'frequent' (default: 5).",
    )
    parser.add_argument(
        "--examples-per-token",
        type=int,
        default=10,
        help="Representative example windows per frequent token (default: 10).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    if not args.windows_jsonl.is_file():
        print(f"error: windows JSONL not found: {args.windows_jsonl}", file=sys.stderr)
        return 1
    assignments_path = args.assignments_csv or args.assignments_json
    if assignments_path is None or not Path(assignments_path).is_file():
        print(f"error: assignments file not found: {assignments_path}", file=sys.stderr)
        return 1

    windows = load_windows(args.windows_jsonl)
    assignments = load_assignments(args.assignments_csv, args.assignments_json)
    summary = build_summary(windows, assignments)
    manifest = write_inspection_outputs(
        args.output_dir,
        windows,
        assignments,
        summary,
        min_token_count=args.min_token_count,
        examples_per_token=args.examples_per_token,
    )
    print(
        json.dumps(
            {
                "windows_total": summary["windows_total"],
                "assignments_total": summary["assignments_total"],
                "windows_joined": summary["windows_joined"],
                "num_tokens": summary["num_tokens"],
                **manifest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
