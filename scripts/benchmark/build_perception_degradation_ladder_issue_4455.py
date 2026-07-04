"""Build issue #4455 perception-degradation ladder smoke configs.

The script expands a preregistered ladder manifest into one camera-ready campaign
config per degradation profile. It does not run or submit a benchmark campaign.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.camera_ready._config import load_campaign_config
from robot_sf.benchmark.observation_noise import (
    normalize_observation_noise_spec,
    observation_noise_hash,
)

DEFAULT_MANIFEST = Path("configs/benchmarks/perception_degradation/issue_4455_ladder_v1.yaml")
DEFAULT_OUT_DIR = Path("output/benchmarks/issue_4455_perception_degradation_ladder")


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a YAML mapping")
    return data


def _require_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    values = payload.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} must be a non-empty list")
    if any(not isinstance(value, dict) for value in values):
        raise TypeError(f"{key} entries must be mappings")
    return values


def _campaign_payload(manifest: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    profile_key = str(profile.get("key") or "").strip()
    if not profile_key:
        raise ValueError("profile key is required")
    observation_noise = normalize_observation_noise_spec(profile.get("observation_noise"))
    declared_hash = str(profile.get("profile_hash") or "").strip()
    computed_hash = observation_noise_hash(observation_noise)
    if declared_hash and declared_hash != computed_hash:
        raise ValueError(
            f"profile {profile_key!r} declared hash {declared_hash!r} "
            f"does not match normalized hash {computed_hash!r}"
        )
    defaults = dict(manifest.get("cpu_smoke_defaults") or {})
    return {
        "name": f"issue_4455_perception_degradation_{profile_key}",
        "scenario_matrix": manifest["scenario_matrix"],
        "seed_policy": manifest["seed_policy"],
        "workers": int(defaults.get("workers", 1)),
        "horizon": int(defaults.get("horizon", 5)),
        "dt": float(defaults.get("dt", 0.1)),
        "record_forces": bool(defaults.get("record_forces", False)),
        "resume": bool(defaults.get("resume", False)),
        "export_publication_bundle": bool(defaults.get("export_publication_bundle", False)),
        "observation_noise": observation_noise,
        "planners": manifest["planners"],
        "claim_boundary": manifest["claim_boundary"],
    }


def build_ladder_configs(
    manifest_path: Path = DEFAULT_MANIFEST,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    validate_only: bool = False,
) -> list[Path]:
    """Validate the preregistered ladder and optionally write campaign configs."""

    manifest = _load_yaml(manifest_path)
    if manifest.get("schema_version") != "perception-degradation-ladder.v1":
        raise ValueError("schema_version must be perception-degradation-ladder.v1")
    if int(manifest.get("issue", 0)) != 4455:
        raise ValueError("issue must be 4455")
    if not manifest.get("scenario_matrix"):
        raise ValueError("scenario_matrix is required")
    if not isinstance(manifest.get("seed_policy"), dict):
        raise TypeError("seed_policy must be a mapping")
    _require_list(manifest, "planners")
    profiles = _require_list(manifest, "profiles")

    generated: list[Path] = []
    if not validate_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    for profile in profiles:
        payload = _campaign_payload(manifest, profile)
        if validate_only:
            continue
        path = out_dir / f"{payload['name']}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        load_campaign_config(path)
        generated.append(path)
    return generated


def main() -> int:
    """Run the issue #4455 ladder config builder CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    generated = build_ladder_configs(
        args.manifest,
        args.out_dir,
        validate_only=args.validate_only,
    )
    if args.validate_only:
        print(f"validated {args.manifest}")
    else:
        print(f"wrote {len(generated)} configs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
