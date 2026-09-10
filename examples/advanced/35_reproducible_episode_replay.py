"""Compare same-seed episode records and a changed-seed input.

Usage:
    uv run python examples/advanced/35_reproducible_episode_replay.py
    uv run python examples/advanced/35_reproducible_episode_replay.py \
        --output-dir output/example-replay --format json

Expected Output:
    - Three ``EpisodeRecord`` JSON files and a versioned comparison report.
    - ``identical`` for the repeated seed and ``different`` for the changed seed.

Limitations:
    - Same-host exact replay is a local diagnostic, not a cross-host or platform
      determinism claim.
    - A changed seed identifies a changed input; this example does not attribute
      the outcome to a particular stochastic factor.
    - The report is smoke evidence only and does not support a benchmark claim.

References:
    - docs/glossary.md
    - https://gymnasium.farama.org/api/env/
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_sf import EpisodeRecord, make_env, run_episode
from robot_sf.benchmark.types import (
    CANONICAL_EPISODE_RECORD_FIELDS,
    CANONICAL_EPISODE_RUNTIME_FIELDS,
    CANONICAL_EPISODE_RUNTIME_METRICS,
)

SCHEMA_VERSION = "episode_replay_comparison.v1"
SCENARIO = "quickstart_demo"
SAME_SEED = 118
DIFFERENT_SEED = 119
DEFAULT_MAX_STEPS = 4
DEFAULT_OUTPUT_DIR = Path("output/example-replay")


def _max_steps_from_environment() -> int:
    """Resolve the bounded example horizon from the standard smoke override."""
    value = os.environ.get("ROBOT_SF_EXAMPLES_MAX_STEPS")
    if value:
        try:
            return max(1, int(value))
        except ValueError:
            pass
    return DEFAULT_MAX_STEPS


def _run_record(seed: int, max_steps: int) -> EpisodeRecord:
    """Run one public-facade episode and close its environment."""
    env = make_env(scenario=SCENARIO, seed=seed)
    try:
        return run_episode(env, max_steps=max_steps, seed=seed)
    finally:
        env.close()


def _identity(record: EpisodeRecord) -> dict[str, Any]:
    """Return the input-bound identity shown in the comparison report."""
    return {
        "episode_id": record.episode_id,
        "scenario_id": record.scenario_id,
        "seed": record.seed,
    }


def _first_difference(left: Any, right: Any, path: str = "record") -> list[str]:
    """Return exact paths where two canonical JSON values differ."""
    if type(left) is not type(right):
        return [path]
    if isinstance(left, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}.{key}"
            if key not in left or key not in right:
                paths.append(child_path)
            else:
                paths.extend(_first_difference(left[key], right[key], child_path))
        return paths
    if isinstance(left, list):
        paths = []
        if len(left) != len(right):
            paths.append(path)
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=False)):
            paths.extend(_first_difference(left_item, right_item, f"{path}[{index}]"))
        return paths
    return [] if left == right else [path]


def _comparison(
    first: EpisodeRecord,
    second: EpisodeRecord,
    first_digest: str,
    second_digest: str,
) -> dict[str, Any]:
    """Build one comparison row with identity and canonical-digest evidence."""
    first_payload = first.canonical_payload()
    second_payload = second.canonical_payload()
    identity_match = _identity(first) == _identity(second)
    digest_match = first_digest == second_digest
    if identity_match and digest_match:
        status = "identical"
    elif identity_match:
        status = "not_comparable"
    else:
        status = "different"
    return {
        "status": status,
        "identity_match": identity_match,
        "digest_match": digest_match,
        "first_identity": _identity(first),
        "second_identity": _identity(second),
        "first_digest": first_digest,
        "second_digest": second_digest,
        "mismatch_paths": _first_difference(first_payload, second_payload),
    }


def _canonical_policy() -> dict[str, Any]:
    """Describe the shared ``EpisodeRecord`` canonicalization policy."""
    return {
        "helper": "robot_sf.benchmark.types.EpisodeRecord.canonical_payload",
        "stable_fields": list(CANONICAL_EPISODE_RECORD_FIELDS),
        "excluded_runtime_fields": sorted(CANONICAL_EPISODE_RUNTIME_FIELDS),
        "excluded_runtime_metrics": sorted(CANONICAL_EPISODE_RUNTIME_METRICS),
        "interpretation": (
            "Timing and raw implementation diagnostics are excluded by the shared record "
            "policy; no example-specific ignore list is applied."
        ),
    }


def build_replay_report(output_dir: Path, max_steps: int) -> dict[str, Any]:
    """Run the three bounded episodes and write records plus a comparison report."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = (
        ("same_seed_a", SAME_SEED),
        ("same_seed_b", SAME_SEED),
        ("different_seed", DIFFERENT_SEED),
    )
    records: dict[str, EpisodeRecord] = {}
    artifacts: list[dict[str, Any]] = []
    for label, seed in cases:
        record = _run_record(seed, max_steps)
        records[label] = record
        record_path = output_dir / f"{label}.json"
        record.save(record_path)
        artifacts.append(
            {
                "label": label,
                "seed": seed,
                "path": record_path.name,
                "episode_id": record.episode_id,
                "scenario_id": record.scenario_id,
                "canonical_digest": record.canonical_digest(),
            }
        )

    same_seed = _comparison(
        records["same_seed_a"],
        records["same_seed_b"],
        records["same_seed_a"].canonical_digest(),
        records["same_seed_b"].canonical_digest(),
    )
    different_seed = _comparison(
        records["same_seed_a"],
        records["different_seed"],
        records["same_seed_a"].canonical_digest(),
        records["different_seed"].canonical_digest(),
    )
    if different_seed["status"] == "different":
        different_seed["interpretation"] = (
            "The seed changed the input identity; no causal explanation for the outcome is made."
        )
    else:
        different_seed["interpretation"] = (
            "The seed input did not produce a distinct identity; classify this as not_comparable "
            "rather than infer a causal seed effect."
        )

    overall_pass = same_seed["status"] == "identical" and different_seed["status"] == "different"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if overall_pass else "not_comparable",
        "evidence_tier": "smoke",
        "claim_boundary": (
            "same-host local replay diagnostic; no cross-host determinism, benchmark, or causal "
            "seed claim"
        ),
        "execution": {
            "scenario": SCENARIO,
            "max_steps": max_steps,
            "same_seed": SAME_SEED,
            "different_seed": DIFFERENT_SEED,
        },
        "canonical_policy": _canonical_policy(),
        "artifacts": artifacts,
        "comparisons": {"same_seed": same_seed, "different_seed": different_seed},
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the bounded example command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=_max_steps_from_environment())
    parser.add_argument("--format", choices=("json", "text"), default="text")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the replay comparison and return a process status."""
    args = _parse_args(argv)
    report = build_replay_report(args.output_dir, args.max_steps)
    same_seed = report["comparisons"]["same_seed"]
    different_seed = report["comparisons"]["different_seed"]
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"same-seed comparison: {same_seed['status']}")
        print(f"changed-seed comparison: {different_seed['status']}")
        print(f"comparison report: {args.output_dir / 'comparison.json'}")
    return 0 if report["status"] == "pass" and different_seed["status"] == "different" else 1


if __name__ == "__main__":
    raise SystemExit(main())
