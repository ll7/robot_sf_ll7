"""Check known masked hybrid tuning rules without changing runtime configuration.

Run with ``python -m scripts.validation.check_hybrid_config_consumption CONFIG``.
This opt-in preflight targets map-backed continuous geometry, not every possible
unused parameter or a simulation of candidate eligibility (issue #10006).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.policy_search_manifest import (
    is_candidate_manifest,
    resolve_candidate_manifest_runtime,
)
from robot_sf.planner.hybrid_rule_local_planner import build_hybrid_rule_local_planner_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MASKED_RULES = ("static_clearance_escape_enabled", "static_corridor_transit_enabled")


def _read_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: planner configuration must be a mapping")
    return data


def _contexts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate declared override contexts, retaining runtime family precedence."""
    contexts: list[dict[str, Any]] = [{"name": "default"}]
    for key in ("scenario_overrides", "scenario_algo_overrides", "family_overrides"):
        overrides = manifest.get(key, {})
        if not isinstance(overrides, dict):
            raise ValueError(f"{key} must be a mapping")
        for name, value in overrides.items():
            if not isinstance(name, str) or not isinstance(value, dict):
                raise ValueError(f"{key} entries must map names to mappings")
            contexts.append(
                {"name": f"family:{name}", "family": name}
                if key == "family_overrides"
                else {"name": name}
            )
    return contexts


def _load_base(path: Path, raw_path: object) -> dict[str, Any]:
    """Read a runtime base with manifest-local then repository-root resolution."""
    if raw_path is None:
        return {}
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("base_config_path must be a nonempty path")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        local = path.parent / candidate
        candidate = local if local.exists() else REPO_ROOT / candidate
    base = _read_mapping(candidate)
    if is_candidate_manifest(base):
        raise ValueError(f"{candidate}: nested candidate bases are not supported")
    return base


def check_config(path: Path) -> list[str]:
    """Return masked-rule findings for default, family and scenario contexts.

    Uses the same override resolver and boolean parser as the runtime. The
    preflight assumes exact map geometry is available; a geometry-free fallback
    can still reach these rules and is outside this check's declared scope.
    """
    manifest = _read_mapping(path)
    problems = []
    for scenario in _contexts(manifest):
        algo, effective = resolve_candidate_manifest_runtime(
            default_algo=str(manifest.get("algo", "hybrid_rule_local_planner")),
            manifest=manifest,
            scenario=scenario,
            load_config=lambda raw_path: _load_base(path, raw_path),
        )
        if algo != "hybrid_rule_local_planner":
            raise ValueError(f"{scenario['name']}: unsupported algorithm {algo!r}")
        config = build_hybrid_rule_local_planner_config(effective)
        if not (
            config.continuous_static_clearance_enabled
            and config.corridor_subgoal_use_continuous_static_check
        ):
            continue
        for key in MASKED_RULES:
            if getattr(config, key):
                problems.append(
                    f"{path}:{scenario['name']}: {key} is unreachable on the configured "
                    "continuous-geometry path; disable the masked rule or select its coarse path"
                )
    return list(dict.fromkeys(problems))


def main() -> int:
    """Print actionable findings and fail closed on malformed/unsupported inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()
    problems = []
    try:
        for path in args.configs:
            problems.extend(check_config(path))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
