#!/usr/bin/env python3
"""Compare preserved calibration rows without running environments or planners."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np


def canonical_metric_value(value):
    """Retain binary float identity, including signed zero and nonfinite sentinels."""
    if isinstance(value, float):
        return {"float64_hex": value.hex()}
    if isinstance(value, list):
        return [canonical_metric_value(item) for item in value]
    if isinstance(value, dict):
        return {key: canonical_metric_value(item) for key, item in value.items()}
    return value


def metric_row_sha256(row: dict) -> str:
    """Hash every metric column, step count and navigation status; exclude provenance/time."""
    payload = {key: row[key] for key in ("metrics", "metric_values", "steps", "status")}
    data = json.dumps(
        canonical_metric_value(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(data).hexdigest()


def load_rows(root: Path) -> tuple[dict, dict]:
    """Require the exact 14-arm, 48-cell, two-development-seed grid and hash raw inputs."""
    rows, files = {}, {}
    for path in sorted(root.glob("runs/*/episodes.jsonl")):
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        arm = path.parent.name.removesuffix("__differential_drive")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = (arm, row["scenario_id"], row["seed"])
            if key in rows or type(key[2]) is not int or key[2] not in {1001, 1002}:
                raise ValueError("duplicate row or non-development seed")
            rows[key] = row
    counts = Counter(key[0] for key in rows)
    cells = {(key[1], key[2]) for key in rows}
    if len(files) != 14 or len(rows) != 1344 or set(counts.values()) != {96} or len(cells) != 96:
        raise ValueError("incomplete 14 x 48 x 2 development grid")
    return rows, files


def compare_rows(left: dict, right: dict) -> dict:
    """Keep all paired row hashes and every differing row's steps/status/metric fields."""
    if left.keys() != right.keys():
        raise ValueError("comparison grids differ")
    hashes, differences = [], []
    for key in sorted(left):
        a, b = left[key], right[key]
        ha, hb = metric_row_sha256(a), metric_row_sha256(b)
        identity = {"arm": key[0], "scenario": key[1], "seed": key[2]}
        hashes.append({**identity, "left_sha256": ha, "right_sha256": hb, "identical": ha == hb})
        if ha != hb:
            changed = []
            for group in ("metrics", "metric_values"):
                for field in sorted(set(a[group]) | set(b[group])):
                    # Presence is significant, including missing versus an explicit null.
                    if (field in a[group]) != (field in b[group]) or canonical_metric_value(
                        a[group].get(field)
                    ) != canonical_metric_value(b[group].get(field)):
                        changed.append(f"{group}.{field}")
            differences.append(
                {
                    **identity,
                    "left": {"steps": a["steps"], "status": a["status"]},
                    "right": {"steps": b["steps"], "status": b["status"]},
                    "changed_metric_columns": changed,
                    "anchor_metrics": {
                        field: {"left": a["metrics"][field], "right": b["metrics"][field]}
                        for field in ("robot_force_impulse_total", "jerk_mean", "curvature_mean")
                    },
                }
            )
    return {
        "rows": len(hashes),
        "identical_rows": len(hashes) - len(differences),
        "different_rows": len(differences),
        "different_rows_by_arm": dict(sorted(Counter(r["arm"] for r in differences).items())),
        "step_differences": sum(r["left"]["steps"] != r["right"]["steps"] for r in differences),
        "status_differences": sum(r["left"]["status"] != r["right"]["status"] for r in differences),
        "row_hashes": hashes,
        "differences": differences,
    }


def environment(root: Path, *, portable: bool) -> dict:
    """Read the recorded CPU/software/thread context; public output obscures node names."""
    context = json.loads((root / "run_meta.json").read_text())["execution_context"]
    if portable:
        context = dict(context)
        context["node_identity_sha256"] = hashlib.sha256(
            context.pop("hostname").encode()
        ).hexdigest()
    return context


def upper_anchors(rows: dict) -> dict:
    """Independently reduce the three recorded scalar columns with linear p95."""
    return {
        term: float(
            np.percentile([row["metrics"][field] for row in rows.values()], 95, method="linear")
        )
        for term, field in (
            ("F", "robot_force_impulse_total"),
            ("J", "jerk_mean"),
            ("K", "curvature_mean"),
        )
    }


def jerk_percentile_details(rows: dict, original_p95: float) -> dict:
    """Explain interpolation and tied sample ranks instead of seeking p95 among raw values."""
    ordered = sorted((row["metrics"]["jerk_mean"], *key) for key, row in rows.items())
    index = (len(ordered) - 1) * 0.95
    return {
        "zero_based_index": index,
        "bracketing_rows": ordered[int(index) : int(index) + 2],
        "p95": float(np.percentile([row[0] for row in ordered], 95, method="linear")),
        "rows_above_original_p95": sum(row[0] > original_p95 for row in ordered),
    }


def classify_repeat(different_rows: int, same_environment: bool) -> str:
    """Require the fixed recorded environment before either scientific a/b classification."""
    if not same_environment:
        return "unresolved_environment"
    return "b" if different_rows else "a"


def build_receipt(original: Path, repeat: Path, rehearsal: Path, *, portable: bool = True) -> dict:
    """Emit measured differences without asserting a causal mechanism or scientific approval."""
    roots = {"original": original, "repeat": repeat, "rehearsal": rehearsal}
    rows, raw, contexts = {}, {}, {}
    for label, root in roots.items():
        rows[label], raw[label] = load_rows(root)
        contexts[label] = environment(root, portable=portable)
    comparison = compare_rows(rows["original"], rows["repeat"])
    previous = compare_rows(rows["rehearsal"], rows["original"])
    same_environment = contexts["original"] == contexts["repeat"]
    classification = classify_repeat(comparison["different_rows"], same_environment)
    anchors = {label: upper_anchors(data) for label, data in rows.items()}
    return {
        "schema": "snqi-v2-determinism-review-input.v1",
        "scientific_review": "pending_independent_review",
        "hash_rule": "SHA256(compact sorted JSON of metrics, metric_values, steps, status; float leaves encoded as {float64_hex:float.hex()}); excludes timestamps/wall_time/provenance",
        "classification": classification,
        "same_recorded_environment": same_environment,
        "execution_contexts": contexts,
        "episode_files_sha256": raw,
        "upper_anchors": anchors,
        "jerk_percentile_details": {
            label: jerk_percentile_details(data, anchors["original"]["J"])
            for label, data in rows.items()
        },
        "rehearsal_to_original_delta": {
            term: anchors["original"][term] - anchors["rehearsal"][term] for term in ("F", "J", "K")
        },
        "J_delta_percent": (anchors["original"]["J"] / anchors["rehearsal"]["J"] - 1) * 100,
        "original_vs_repeat": comparison,
        "rehearsal_vs_original": previous,
        "interpretation_limit": "Fixed recorded environment is tested once. Cross-environment differences are measured, not causally isolated.",
    }
