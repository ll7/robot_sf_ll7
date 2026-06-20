#!/usr/bin/env python3
"""Build a mixed predictive-planner dataset with hard-case oversampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.planner.obstacle_features import (
    PREDICTIVE_EGO_FEATURE_DIM,
    PREDICTIVE_EGO_FEATURE_SCHEMA,
    PREDICTIVE_OBSTACLE_FEATURE_DIM,
    predictive_ego_motion_channel_producer_key,
)

DEFAULT_HARDCASE_REPEAT = 2
DEFAULT_SHUFFLE_SEED = 42
DEFAULT_WEIGHTING_PROFILE_ID = "cli_hardcase_repeat_v1"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset mixing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--hardcase-dataset", type=Path, required=True)
    parser.add_argument("--hardcase-repeat", type=int)
    parser.add_argument(
        "--weighting-spec",
        type=Path,
        help=(
            "Optional YAML/JSON spec describing the hard-case weighting profile. "
            "Explicit --hardcase-repeat and --shuffle-seed values override spec values."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/tmp/predictive_planner/datasets/predictive_rollouts_mixed_v1.npz"),
    )
    parser.add_argument("--shuffle-seed", type=int)
    return parser.parse_args()


def _read_weighting_spec(path: Path) -> dict[str, Any]:
    """Read a hard-case weighting spec from YAML or JSON and return a mapping."""
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Weighting spec must be a mapping: {path}")
    return payload


def _resolve_weighting_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve CLI/spec weighting into one deterministic profile summary."""
    spec_path = getattr(args, "weighting_spec", None)
    spec = _read_weighting_spec(spec_path) if spec_path is not None else {}
    weighting = spec.get("weighting", {})
    if not isinstance(weighting, dict):
        raise TypeError(f"weighting must be a mapping in weighting spec: {spec_path}")

    rule = str(weighting.get("rule") or "repeat_hardcase_rows")
    if rule != "repeat_hardcase_rows":
        raise ValueError(
            f"Unsupported hard-case weighting rule {rule!r}; expected 'repeat_hardcase_rows'."
        )

    hardcase_repeat_raw = (
        args.hardcase_repeat
        if getattr(args, "hardcase_repeat", None) is not None
        else weighting.get("hardcase_repeat", DEFAULT_HARDCASE_REPEAT)
    )
    shuffle_seed_raw = (
        args.shuffle_seed
        if getattr(args, "shuffle_seed", None) is not None
        else weighting.get("shuffle_seed", DEFAULT_SHUFFLE_SEED)
    )
    hardcase_repeat = int(hardcase_repeat_raw)
    shuffle_seed = int(shuffle_seed_raw)
    if hardcase_repeat < 1:
        raise ValueError("--hardcase-repeat must be >= 1")

    profile_id = str(
        spec.get("profile_id") or weighting.get("profile_id") or DEFAULT_WEIGHTING_PROFILE_ID
    )
    hardcase_family = str(weighting.get("hardcase_family") or "unspecified")
    claim_boundary = str(
        spec.get("claim_boundary")
        or "launch/config/tooling only; no retrained checkpoint, hard-seed benchmark evidence, "
        "or model-improvement claim"
    )
    return {
        "profile_id": profile_id,
        "spec_path": str(spec_path) if spec_path is not None else None,
        "rule": rule,
        "hardcase_repeat": hardcase_repeat,
        "hardcase_family": hardcase_family,
        "shuffle_seed": shuffle_seed,
        "claim_boundary": claim_boundary,
        "source": "weighting_spec" if spec_path is not None else "cli_defaults",
    }


def _load_optional_feature_schema_metadata(raw: Any, *, path: Path) -> dict[str, Any] | None:
    """Return optional predictive feature-schema metadata embedded in an NPZ payload."""
    if "feature_schema_json" not in raw:
        return None
    raw_value = raw["feature_schema_json"]
    if isinstance(raw_value, np.ndarray) and raw_value.shape == ():
        raw_value = raw_value.item()
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    try:
        if isinstance(raw_value, str):
            payload = json.loads(raw_value)
            if isinstance(payload, dict):
                return payload
        raise ValueError("Missing or invalid feature_schema_json in dataset payload.")
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid feature_schema_json in dataset {path}: {exc}") from exc


def _is_ego_conditioned_schema(
    feature_schema: dict[str, Any] | None,
    *,
    input_dim: int,
) -> bool:
    """Return whether a dataset width/schema uses ego-conditioned predictive slots."""
    if int(input_dim) == PREDICTIVE_EGO_FEATURE_DIM:
        return True
    if int(input_dim) == PREDICTIVE_EGO_FEATURE_DIM + PREDICTIVE_OBSTACLE_FEATURE_DIM:
        return True
    if not isinstance(feature_schema, dict):
        return False
    schema_name = str(feature_schema.get("name") or "").strip()
    base_schema = str(feature_schema.get("base_schema") or "").strip()
    return PREDICTIVE_EGO_FEATURE_SCHEMA in {schema_name, base_schema}


def _feature_schema_contract(feature_schema: dict[str, Any]) -> dict[str, Any]:
    """Return the schema fields that must agree across mixed dataset inputs."""
    return {
        "name": feature_schema.get("name"),
        "base_schema": feature_schema.get("base_schema"),
        "base_feature_dim": feature_schema.get("base_feature_dim"),
        "input_dim": feature_schema.get("input_dim"),
        "obstacle_feature_schema": feature_schema.get("obstacle_feature_schema"),
    }


def _validate_schema_width(
    *,
    role: str,
    path: Path,
    feature_schema: dict[str, Any],
    input_dim: int,
) -> None:
    """Fail when embedded schema metadata disagrees with the actual feature width."""
    try:
        declared_dim = int(feature_schema.get("input_dim", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Predictive feature schema for {role}={path} has invalid input_dim "
            f"{feature_schema.get('input_dim')!r}."
        ) from exc
    if declared_dim != int(input_dim):
        raise ValueError(
            "Predictive feature schema input_dim mismatch for "
            f"{role}={path}: metadata input_dim={declared_dim}, "
            f"array width={int(input_dim)}"
        )


def _resolve_mixed_feature_schema(
    *,
    base_path: Path,
    hardcase_path: Path,
    base_schema: dict[str, Any] | None,
    hardcase_schema: dict[str, Any] | None,
    base_input_dim: int,
    hardcase_input_dim: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return compatible mixed feature-schema metadata or fail on new ambiguous ego datasets."""
    base_is_ego = _is_ego_conditioned_schema(base_schema, input_dim=base_input_dim)
    hardcase_is_ego = _is_ego_conditioned_schema(hardcase_schema, input_dim=hardcase_input_dim)
    if base_schema is None or hardcase_schema is None:
        if base_is_ego or hardcase_is_ego:
            missing = []
            if base_schema is None:
                missing.append(f"base={base_path}")
            if hardcase_schema is None:
                missing.append(f"hardcase={hardcase_path}")
            raise ValueError(
                "Ego-conditioned mixed predictive datasets require feature_schema_json on both "
                "inputs; missing metadata for " + ", ".join(missing)
            )
        return None, None

    _validate_schema_width(
        role="base",
        path=base_path,
        feature_schema=base_schema,
        input_dim=base_input_dim,
    )
    _validate_schema_width(
        role="hardcase",
        path=hardcase_path,
        feature_schema=hardcase_schema,
        input_dim=hardcase_input_dim,
    )

    if _feature_schema_contract(base_schema) != _feature_schema_contract(hardcase_schema):
        raise ValueError(
            "Predictive feature schema mismatch between mixed dataset inputs: "
            f"base={_feature_schema_contract(base_schema)} "
            f"hardcase={_feature_schema_contract(hardcase_schema)}"
        )

    base_producer = predictive_ego_motion_channel_producer_key(base_schema)
    hardcase_producer = predictive_ego_motion_channel_producer_key(hardcase_schema)
    if (
        base_producer is not None
        and hardcase_producer is not None
        and base_producer != hardcase_producer
    ):
        raise ValueError(
            "Predictive ego motion producer mismatch between mixed dataset inputs: "
            f"base={base_producer!r} hardcase={hardcase_producer!r}"
        )
    if base_is_ego or hardcase_is_ego:
        if base_producer is None or hardcase_producer is None:
            raise ValueError(
                "Ego-conditioned mixed predictive datasets require ego_motion_channel_producer "
                "metadata on both inputs."
            )
    return dict(base_schema), {"producer_key": base_producer or hardcase_producer}


def _load_npz(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    """Load arrays plus optional feature-schema metadata from a dataset NPZ."""
    with np.load(path) as raw:
        state = np.asarray(raw["state"], dtype=np.float32)
        target = np.asarray(raw["target"], dtype=np.float32)
        mask = np.asarray(raw["mask"], dtype=np.float32)
        target_mask = (
            np.asarray(raw["target_mask"], dtype=np.float32)
            if "target_mask" in raw
            else np.repeat(mask[:, :, None], target.shape[2], axis=2).astype(np.float32)
        )
        feature_schema = _load_optional_feature_schema_metadata(raw, path=path)
    return state, target, mask, target_mask, feature_schema


def main() -> int:
    """Create mixed dataset and write metadata sidecar."""
    args = parse_args()
    weighting_profile = _resolve_weighting_profile(args)

    base_state, base_target, base_mask, base_target_mask, base_feature_schema = _load_npz(
        args.base_dataset
    )
    hard_state, hard_target, hard_mask, hard_target_mask, hard_feature_schema = _load_npz(
        args.hardcase_dataset
    )

    for arr_base, arr_hard, name in [
        (base_state, hard_state, "state"),
        (base_target, hard_target, "target"),
        (base_mask, hard_mask, "mask"),
        (base_target_mask, hard_target_mask, "target_mask"),
    ]:
        if arr_base.shape[1:] != arr_hard.shape[1:]:
            raise ValueError(
                f"Shape mismatch for {name}: base {arr_base.shape} vs hardcase {arr_hard.shape}"
            )

    mixed_feature_schema, producer_summary = _resolve_mixed_feature_schema(
        base_path=args.base_dataset,
        hardcase_path=args.hardcase_dataset,
        base_schema=base_feature_schema,
        hardcase_schema=hard_feature_schema,
        base_input_dim=int(base_state.shape[2]),
        hardcase_input_dim=int(hard_state.shape[2]),
    )
    feature_compatibility = {
        "status": "compatible",
        "base_input_dim": int(base_state.shape[2]),
        "hardcase_input_dim": int(hard_state.shape[2]),
        "feature_schema_required": bool(
            _is_ego_conditioned_schema(base_feature_schema, input_dim=int(base_state.shape[2]))
            or _is_ego_conditioned_schema(
                hard_feature_schema,
                input_dim=int(hard_state.shape[2]),
            )
        ),
        "feature_schema_present": {
            "base": base_feature_schema is not None,
            "hardcase": hard_feature_schema is not None,
        },
        "feature_schema_contract": (
            _feature_schema_contract(mixed_feature_schema)
            if mixed_feature_schema is not None
            else None
        ),
        "ego_motion_channel_producer": producer_summary,
    }
    feature_schema_json = (
        json.dumps(mixed_feature_schema, sort_keys=True)
        if mixed_feature_schema is not None
        else None
    )

    hard_rep = int(weighting_profile["hardcase_repeat"])
    state = np.concatenate([base_state, np.repeat(hard_state, hard_rep, axis=0)], axis=0)
    target = np.concatenate([base_target, np.repeat(hard_target, hard_rep, axis=0)], axis=0)
    mask = np.concatenate([base_mask, np.repeat(hard_mask, hard_rep, axis=0)], axis=0)
    target_mask = np.concatenate(
        [base_target_mask, np.repeat(hard_target_mask, hard_rep, axis=0)],
        axis=0,
    )

    rng = np.random.default_rng(int(weighting_profile["shuffle_seed"]))
    idx = np.arange(state.shape[0])
    rng.shuffle(idx)
    state = state[idx]
    target = target[idx]
    mask = mask[idx]
    target_mask = target_mask[idx]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "state": state,
        "target": target,
        "mask": mask,
        "target_mask": target_mask,
    }
    if feature_schema_json is not None:
        payload["feature_schema_json"] = feature_schema_json
    np.savez_compressed(
        args.output,
        **payload,
    )

    summary = {
        "base_dataset": str(args.base_dataset),
        "hardcase_dataset": str(args.hardcase_dataset),
        "base_count": int(base_state.shape[0]),
        "hard_case_count": int(hard_state.shape[0]),
        "output_count": int(state.shape[0]),
        "hardcase_repeat": hard_rep,
        "num_base_samples": int(base_state.shape[0]),
        "num_hardcase_samples": int(hard_state.shape[0]),
        "num_output_samples": int(state.shape[0]),
        "weighting_profile": weighting_profile["profile_id"],
        "weighting_rule": weighting_profile,
        "active_agent_ratio": float(np.mean(mask)),
        "active_target_ratio": float(np.mean(target_mask)),
        "shuffle_seed": int(weighting_profile["shuffle_seed"]),
        "output": str(args.output),
        "feature_compatibility": feature_compatibility,
        "feature_schema": mixed_feature_schema,
        "feature_schema_json": feature_schema_json,
        "ego_motion_channel_producer": producer_summary,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
