#!/usr/bin/env python3
"""Run the bounded BC loader/overfit smoke for the issue #1496 oracle dataset.

This is the executable entry point for :mod:`robot_sf.training.oracle_imitation_bc_smoke`. It
resolves the ``expert_traj_v1.npz`` artifact either directly (``--dataset-path``) or through
the durable trace-URI registry (``--registry`` + ``--private-artifact-root``), enforces the
registry contract and checksum, runs the BC loader/overfit smoke, and prints the result.

Claim boundary: smoke evidence only -- not benchmark or warm-start-quality evidence. See
:mod:`robot_sf.training.oracle_imitation_bc_smoke` for the full boundary.

Usage::

    # Direct artifact path
    uv run python scripts/validation/run_oracle_imitation_bc_smoke.py \\
        --dataset-path /path/to/expert_traj_v1.npz --output-dir output/smoke/issue_1496

    # Resolve + checksum-verify through the durable registry
    uv run python scripts/validation/run_oracle_imitation_bc_smoke.py \\
        --registry configs/training/ppo_imitation/oracle_trace_uri_registry_issue_1496_collection.yaml \\
        --private-artifact-root /path/to/private_artifacts \\
        --output-dir output/smoke/issue_1496
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

import yaml

from robot_sf.benchmark.artifact_catalog import sha256_file
from robot_sf.training.oracle_imitation_bc_smoke import (
    BCSmokeConfig,
    OracleImitationBcSmokeError,
    run_bc_overfit_smoke,
)
from robot_sf.training.oracle_trace_uri_registry import (
    OracleTraceUriRegistryError,
    validate_trace_uri_registry,
)

_PRIVATE_ARTIFACT_URI_PREFIX = "private-artifact://"
_HEX_DIGITS = frozenset("0123456789abcdef")
_SHA256_LENGTH = 64


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the bounded BC loader/overfit smoke for issue #1496."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-path",
        type=Path,
        help="Direct path to the expert_traj_v1.npz artifact.",
    )
    source.add_argument(
        "--registry",
        type=Path,
        help=(
            "Durable trace-URI registry YAML. Requires --private-artifact-root; resolves and "
            "checksum-verifies the dataset artifact before running the smoke."
        ),
    )
    parser.add_argument(
        "--private-artifact-root",
        type=Path,
        help="Root directory for resolving private-artifact:// URIs (required with --registry).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/smoke/issue_1496/oracle_imitation_bc_smoke",
        help="Directory to receive the checkpoint, metrics, and smoke manifest.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of full-batch steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="MLP hidden width.")
    parser.add_argument("--seed", type=int, default=1496, help="Torch/numpy RNG seed.")
    parser.add_argument(
        "--repo-root",
        default=Path.cwd(),
        type=Path,
        help="Repository root used for registry validation (defaults to cwd).",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON result and exit 0.")
    return parser


def _resolve_private_artifact_path(uri: str, private_root: Path) -> Path:
    """Resolve a private-artifact URI beneath its root, jail-checked (no traversal)."""
    relative = uri.removeprefix(_PRIVATE_ARTIFACT_URI_PREFIX)
    segments = relative.split("/")
    if (
        not relative
        or relative.startswith("/")
        or "\\" in relative
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise OracleImitationBcSmokeError(f"invalid private-artifact relative path: {uri}")
    root = private_root.resolve()
    candidate = (root / PurePosixPath(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise OracleImitationBcSmokeError(
            f"private-artifact URI escapes configured root: {uri}"
        ) from exc
    return candidate


def _load_registry(path: Path) -> dict[str, object]:
    """Load a registry YAML mapping with the CLI's fail-closed exception type."""
    try:
        registry = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise OracleImitationBcSmokeError(f"failed to load registry YAML: {path}") from exc
    if not isinstance(registry, dict):
        raise OracleImitationBcSmokeError("trace-URI registry must be a YAML mapping")
    return registry


def _dataset_artifact_metadata(registry: dict[str, object]) -> tuple[str, str]:
    """Return a validated private dataset URI and concrete SHA-256 from a registry."""
    dataset_artifact = registry.get("dataset_artifact")
    if not isinstance(dataset_artifact, dict):
        raise OracleImitationBcSmokeError(
            "registry dataset_artifact must be a mapping with uri and sha256"
        )
    uri = dataset_artifact.get("uri")
    expected_sha = dataset_artifact.get("sha256")
    if not isinstance(uri, str) or not uri.startswith(_PRIVATE_ARTIFACT_URI_PREFIX):
        raise OracleImitationBcSmokeError(
            "registry dataset_artifact.uri must be a private-artifact:// URI when using --registry"
        )
    if not isinstance(expected_sha, str):
        raise OracleImitationBcSmokeError(
            "registry dataset_artifact.sha256 must be a concrete 64-character SHA-256 digest"
        )
    expected_sha = expected_sha.strip().lower()
    if len(expected_sha) != _SHA256_LENGTH or not set(expected_sha) <= _HEX_DIGITS:
        raise OracleImitationBcSmokeError(
            "registry dataset_artifact.sha256 must be a concrete 64-character SHA-256 digest"
        )
    return uri, expected_sha


def _validate_registry_contract(args: argparse.Namespace) -> None:
    """Enforce the full durable trace contract for a registry-backed smoke."""
    try:
        validate_trace_uri_registry(
            args.registry,
            repo_root=args.repo_root,
            private_artifact_root=args.private_artifact_root,
            require_training_ready=True,
        )
    except OracleTraceUriRegistryError as exc:
        raise OracleImitationBcSmokeError(f"registry contract validation failed: {exc}") from exc


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    """Resolve and (when via the registry) checksum-verify the dataset artifact path.

    Raises:
        OracleImitationBcSmokeError: If resolution fails, the checksum mismatches, or the
            registry contract is violated.
    """
    if args.dataset_path is not None:
        return args.dataset_path

    if args.private_artifact_root is None:
        raise OracleImitationBcSmokeError(
            "--registry requires --private-artifact-root to resolve the dataset artifact"
        )
    if not args.private_artifact_root.is_dir():
        raise OracleImitationBcSmokeError(
            f"--private-artifact-root is not a directory: {args.private_artifact_root}"
        )
    registry = _load_registry(args.registry)
    uri, expected_sha = _dataset_artifact_metadata(registry)
    _validate_registry_contract(args)

    dataset_path = _resolve_private_artifact_path(uri, args.private_artifact_root)
    if not dataset_path.is_file():
        raise OracleImitationBcSmokeError(
            f"dataset artifact resolved but missing on disk: {dataset_path}"
        )
    actual = sha256_file(dataset_path)
    if actual != expected_sha:
        raise OracleImitationBcSmokeError(
            f"dataset checksum mismatch: expected {expected_sha}, got {actual}"
        )
    return dataset_path


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the BC smoke, and print a shell-friendly summary."""
    args = build_arg_parser().parse_args(argv)
    try:
        dataset_path = resolve_dataset_path(args)
        result = run_bc_overfit_smoke(
            BCSmokeConfig(
                dataset_path=str(dataset_path),
                output_dir=args.output_dir,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                hidden_dim=args.hidden_dim,
                seed=args.seed,
            )
        )
    except OracleImitationBcSmokeError as exc:
        if args.json:
            print(json.dumps({"status": "failed", "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"BC smoke failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "evidence_tier": result.evidence_tier,
                    "smoke_not_quality": True,
                    "checkpoint_path": result.checkpoint_path,
                    "manifest_path": result.manifest_path,
                    "metrics_path": result.metrics_path,
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "loss_reduction": result.loss_reduction,
                    "num_train_steps": result.num_train_steps,
                    "episode_counts": result.episode_counts,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(
            "issue #1496 BC loader/overfit smoke OK "
            f"(smoke evidence, not quality): "
            f"loss {result.initial_loss:.6g} -> {result.final_loss:.6g} "
            f"(reduction {result.loss_reduction:.6g}, {result.num_train_steps} train steps, "
            f"splits {result.episode_counts})"
        )
        print(f"  checkpoint: {result.checkpoint_path}")
        print(f"  manifest:   {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
