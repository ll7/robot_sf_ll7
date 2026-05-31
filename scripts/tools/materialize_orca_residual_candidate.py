#!/usr/bin/env python3
"""Materialize a run-local ORCA-residual candidate for a trained BC checkpoint."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = REPO_ROOT / "docs/context/policy_search/candidate_registry.yaml"
DEFAULT_CANDIDATE = "orca_residual_guarded_ppo_v0"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as a mapping, failing when the top-level value is not a dict."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return payload


def _resolve_repo_path(raw: str) -> Path:
    """Resolve a possibly repository-relative path against ``REPO_ROOT``."""
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-model-id", required=True)
    parser.add_argument(
        "--policy-model-path",
        type=Path,
        required=True,
        help="Run-local checkpoint path produced by BC pretraining.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--json", action="store_true")
    return parser


def materialize_candidate(
    *,
    policy_model_id: str,
    policy_model_path: Path,
    output_dir: Path,
    registry_path: Path,
    candidate_name: str,
) -> dict[str, Any]:
    """Write a run-local registry and candidate config pinned to a trained policy id."""
    registry_path = registry_path.resolve()
    registry = _load_yaml(registry_path)
    candidates = registry.get("candidates")
    if not isinstance(candidates, dict) or candidate_name not in candidates:
        raise KeyError(f"Candidate {candidate_name!r} not found in {registry_path}")
    entry = dict(candidates[candidate_name])
    config_raw = entry.get("candidate_config_path")
    if not isinstance(config_raw, str) or not config_raw.strip():
        raise ValueError(f"Candidate {candidate_name!r} has no candidate_config_path")

    candidate_config_path = _resolve_repo_path(config_raw).resolve()
    candidate = _load_yaml(candidate_config_path)
    params = candidate.setdefault("params", {})
    if not isinstance(params, dict):
        raise TypeError(f"Candidate params must be a mapping: {candidate_config_path}")
    policy_model_path = policy_model_path.resolve()
    if not policy_model_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {policy_model_path}")
    params["model_id"] = None
    params["model_path"] = str(policy_model_path)

    output_dir = output_dir.resolve()
    candidate_dir = output_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    runtime_candidate_path = candidate_dir / f"{candidate_name}.yaml"
    runtime_registry_path = output_dir / "candidate_registry.yaml"
    manifest_path = output_dir / "materialized_candidate_manifest.json"

    runtime_candidate_path.write_text(
        yaml.safe_dump(candidate, sort_keys=False),
        encoding="utf-8",
    )
    entry["candidate_config_path"] = str(runtime_candidate_path)
    entry["status"] = "implemented"
    entry["training_required"] = False
    entry["materialized_from"] = str(candidate_config_path)
    entry["materialized_policy_model_id"] = policy_model_id
    candidates[candidate_name] = entry
    runtime_registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": "orca-residual-materialized-candidate.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate": candidate_name,
        "source_candidate_config": str(candidate_config_path),
        "runtime_candidate_config": str(runtime_candidate_path),
        "runtime_registry": str(runtime_registry_path),
        "policy_model_id": policy_model_id,
        "policy_model_path": str(policy_model_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    """Run the materializer."""
    args = build_parser().parse_args()
    manifest = materialize_candidate(
        policy_model_id=args.policy_model_id,
        policy_model_path=args.policy_model_path,
        output_dir=args.output_dir,
        registry_path=args.registry,
        candidate_name=args.candidate,
    )
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(manifest["runtime_registry"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
