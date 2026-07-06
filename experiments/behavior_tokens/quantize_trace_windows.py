"""Deterministic quantization of trace-window features into discrete behavior tokens.

Reads the ``windows.jsonl`` produced by ``extract_windows.py``, standardizes the
subset of features that are finite across enough windows, and assigns each valid
window a discrete token id via k-means (scikit-learn when available, otherwise a
NumPy-only deterministic fallback). All randomness is seeded so repeated runs on the
same input yield identical assignments.

Example::

    uv run python experiments/behavior_tokens/quantize_trace_windows.py \\
        --windows-jsonl output/experiments/behavior_tokens/windows.jsonl \\
        --num-tokens 12 \\
        --output-json output/experiments/behavior_tokens/token_assignments.json \\
        --output-csv output/experiments/behavior_tokens/token_assignments.csv

Claim boundary: token ids are exploratory descriptors, not validated labels or
benchmark evidence. A large transformer/model dependency is intentionally avoided.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.behavior_tokens.schemas import (  # noqa: E402
    CLAIM_BOUNDARY,
    FEATURE_NAMES,
    QUANTIZER_SCHEMA_VERSION,
)


def load_windows(path: Path) -> list[dict[str, Any]]:
    """Load window records from a JSONL file."""
    windows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                windows.append(record)
    return windows


def select_feature_columns(windows: list[dict[str, Any]], min_finite_fraction: float) -> list[str]:
    """Return feature names whose finite-value fraction meets ``min_finite_fraction``.

    Selecting columns by finite fraction (rather than imputing) keeps the token space
    grounded in genuinely measured quantities. Windows that lack any selected feature
    are later excluded as invalid rather than back-filled with fabricated values.
    """
    if not windows:
        return []
    selected: list[str] = []
    for name in FEATURE_NAMES:
        finite = sum(
            1
            for w in windows
            if isinstance(w.get("features"), dict) and _is_finite(w["features"].get(name))
        )
        if finite / len(windows) >= min_finite_fraction:
            selected.append(name)
    return selected


def _is_finite(value: Any) -> bool:
    """Return True when ``value`` is a finite number."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_feature_matrix(
    windows: list[dict[str, Any]], columns: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Return the feature matrix for valid windows plus the excluded count.

    A window is *valid* when all selected ``columns`` are finite for it. Returns
    ``(matrix, valid_windows, excluded_count)``.
    """
    rows: list[list[float]] = []
    valid: list[dict[str, Any]] = []
    excluded = 0
    for window in windows:
        feats = window.get("features")
        if not isinstance(feats, dict) or not all(_is_finite(feats.get(c)) for c in columns):
            excluded += 1
            continue
        rows.append([float(feats[c]) for c in columns])
        valid.append(window)
    matrix = np.asarray(rows, dtype=float) if rows else np.empty((0, len(columns)), dtype=float)
    return matrix, valid, excluded


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score standardize columns; return ``(standardized, mean, std)``.

    Columns with zero variance use a std of 1.0 so they contribute no spurious scale.
    """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    safe_std = np.where(std < 1e-12, 1.0, std)
    return (matrix - mean) / safe_std, mean, safe_std


def _numpy_kmeans(
    data: np.ndarray, num_tokens: int, seed: int, max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic NumPy-only k-means (k-means++ style seeded init).

    Returns ``(labels, centers)``. Used when scikit-learn is unavailable.
    """
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    k = min(num_tokens, n)
    # k-means++ initialization for stable, spread-out centers.
    centers = [data[rng.integers(n)]]
    for _ in range(1, k):
        dist_sq = np.min([np.sum((data - c) ** 2, axis=1) for c in centers], axis=0)
        total = dist_sq.sum()
        if total <= 0:
            centers.append(data[rng.integers(n)])
            continue
        probs = dist_sq / total
        centers.append(data[rng.choice(n, p=probs)])
    centers_arr = np.asarray(centers, dtype=float)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centers_arr[None, :, :], axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(new_labels, labels) and _ > 0:
            labels = new_labels
            break
        labels = new_labels
        for j in range(centers_arr.shape[0]):
            members = data[labels == j]
            if members.size:
                centers_arr[j] = members.mean(axis=0)
    return labels, centers_arr


def _canonicalize_labels(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Relabel clusters deterministically by ordering centers lexicographically.

    scikit-learn's KMeans (via OpenMP) may permute cluster *label integers* between
    otherwise-identical runs on degenerate data, even with a fixed ``random_state``.
    Ordering clusters by their (rounded) center vectors makes token ids stable and
    library-independent, so repeated runs and the NumPy fallback agree.
    """
    order = sorted(range(centers.shape[0]), key=lambda i: tuple(np.round(centers[i], 6).tolist()))
    remap = {old: new for new, old in enumerate(order)}
    new_labels = np.array([remap[int(label)] for label in labels], dtype=int)
    return new_labels, centers[order]


def quantize(data: np.ndarray, num_tokens: int, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Cluster standardized ``data`` into ``num_tokens`` tokens.

    Returns ``(labels, centers, algorithm)`` with deterministically canonicalized
    token ids. Prefers scikit-learn KMeans; falls back to a deterministic NumPy
    implementation when scikit-learn is not importable.
    """
    effective_k = min(num_tokens, data.shape[0])
    try:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=effective_k, random_state=seed, n_init=10)
        labels = model.fit_predict(data)
        canon_labels, canon_centers = _canonicalize_labels(
            labels.astype(int), model.cluster_centers_
        )
        return canon_labels, canon_centers, "sklearn.KMeans"
    except ImportError:
        labels, centers = _numpy_kmeans(data, effective_k, seed)
        canon_labels, canon_centers = _canonicalize_labels(labels, centers)
        return canon_labels, canon_centers, "numpy_kmeans_fallback"


def _write_outputs(
    output_json: Path | None,
    output_csv: Path | None,
    assignments: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Write token assignments to JSON and/or CSV."""
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump(
                {"metadata": metadata, "assignments": assignments},
                handle,
                indent=2,
                sort_keys=True,
            )
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["window_id", "token_id"])
            for item in assignments:
                writer.writerow([item["window_id"], item["token_id"]])


def run_quantization(
    windows: list[dict[str, Any]],
    *,
    num_tokens: int,
    seed: int,
    min_finite_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Quantize windows and return ``(assignments, metadata)``.

    Fails closed by raising ``ValueError`` when no finite feature columns or no valid
    windows remain, rather than emitting empty or fabricated token assignments.
    """
    columns = select_feature_columns(windows, min_finite_fraction)
    if not columns:
        raise ValueError(
            "no feature columns meet the finite-fraction threshold; cannot quantize "
            "(reduce --min-finite-fraction or extract richer traces)"
        )
    matrix, valid_windows, excluded = build_feature_matrix(windows, columns)
    if matrix.shape[0] == 0:
        raise ValueError("no windows have all selected features finite; cannot assign tokens")
    standardized, mean, std = standardize(matrix)
    labels, centers, algorithm = quantize(standardized, num_tokens, seed)

    assignments = [
        {"window_id": window["window_id"], "token_id": int(label)}
        for window, label in zip(valid_windows, labels, strict=True)
    ]
    metadata = {
        "quantizer_schema_version": QUANTIZER_SCHEMA_VERSION,
        "algorithm": algorithm,
        "seed": seed,
        "num_tokens_requested": num_tokens,
        "num_tokens_effective": int(centers.shape[0]),
        "feature_columns": columns,
        "min_finite_fraction": min_finite_fraction,
        "normalization": {
            "mean": {col: float(mean[i]) for i, col in enumerate(columns)},
            "std": {col: float(std[i]) for i, col in enumerate(columns)},
        },
        "token_centers_standardized": {
            str(token): {col: float(centers[token, i]) for i, col in enumerate(columns)}
            for token in range(centers.shape[0])
        },
        "windows_total": len(windows),
        "windows_valid": len(valid_windows),
        "windows_excluded_missing_features": excluded,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return assignments, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for token quantization."""
    parser = argparse.ArgumentParser(
        description=(
            "Quantize trace-window feature vectors into deterministic discrete behavior "
            "tokens. Offline and read-only; no transformer/model dependency."
        ),
        epilog=CLAIM_BOUNDARY,
    )
    parser.add_argument(
        "--windows-jsonl",
        type=Path,
        required=True,
        help="Input windows JSONL from extract_windows.py.",
    )
    parser.add_argument(
        "--num-tokens", type=int, default=12, help="Target discrete vocabulary size (default: 12)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for deterministic clustering (default: 0)."
    )
    parser.add_argument(
        "--min-finite-fraction",
        type=float,
        default=0.9,
        help="Minimum finite fraction for a feature column to be used (default: 0.9).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/experiments/behavior_tokens/token_assignments.json"),
        help="Output JSON path (assignments + metadata).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV output path (window_id, token_id).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    if args.num_tokens <= 0:
        print("error: --num-tokens must be positive", file=sys.stderr)
        return 2
    if not args.windows_jsonl.is_file():
        print(f"error: windows JSONL not found: {args.windows_jsonl}", file=sys.stderr)
        return 1

    windows = load_windows(args.windows_jsonl)
    if not windows:
        print("error: no window records in input JSONL", file=sys.stderr)
        return 1
    try:
        assignments, metadata = run_quantization(
            windows,
            num_tokens=args.num_tokens,
            seed=args.seed,
            min_finite_fraction=args.min_finite_fraction,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _write_outputs(args.output_json, args.output_csv, assignments, metadata)
    print(
        json.dumps(
            {
                "windows_total": metadata["windows_total"],
                "windows_valid": metadata["windows_valid"],
                "windows_excluded_missing_features": metadata["windows_excluded_missing_features"],
                "num_tokens_effective": metadata["num_tokens_effective"],
                "algorithm": metadata["algorithm"],
                "feature_columns": metadata["feature_columns"],
                "output_json": str(args.output_json) if args.output_json else None,
                "output_csv": str(args.output_csv) if args.output_csv else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
