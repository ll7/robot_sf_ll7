#!/usr/bin/env python3
"""Check the issue #5303 search-promotion timing-control contract (read-only).

Validates that the frozen adversarial candidate timing dimensions ``spawn_time_s`` and
``pedestrian_delay_s`` are bound to a concrete pedestrian and change the effective runtime
scenario and its canonical hash, and that the side-effect-free promotion preflight fails
closed when a dimension targets no pedestrian. It runs no search, planner execution, replay,
campaign, or outcome inspection, and it authorizes no promotion campaign.

By default the script runs a built-in contract self-test: it materializes a promotion-ready
search space (with a declared ``pedestrian.id``) that must reach ``promotion_timing_ready``,
and an inert search space (no ``pedestrian.id``) that must be rejected with
``blocked_no_pedestrian`` -- the exact failure mode found in the PR #6291 exact-head review.
It exits non-zero only when the gate misbehaves.

Examples:
    # Built-in contract self-test (positive ready probe + negative rejection control).
    uv run python scripts/tools/check_issue_5303_search_promotion_contract.py

    # Probe a specific search-space / scenario-template pair, JSON output, fail when blocked.
    uv run python scripts/tools/check_issue_5303_search_promotion_contract.py \
        --search-space configs/adversarial/issue_5303_space.yaml \
        --scenario-template configs/scenarios/example.yaml \
        --format json --fail-on-blocked
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf.adversarial.config import RangeConfig, SearchSpaceConfig
from robot_sf.benchmark.issue_5303_search_promotion_preflight import (
    SearchPromotionPreflight,
    SearchPromotionPreflightError,
    evaluate_preflight,
    evaluate_preflight_from_files,
    render_markdown,
    to_dict,
)

#: Pedestrian identity used by the built-in promotion-ready contract probe.
CONTRACT_PEDESTRIAN_ID = "issue_5303_contract_probe"

#: Minimal in-memory scenario template used by the built-in contract self-test so the gate
#: can be exercised without touching on-disk configs.
CONTRACT_TEMPLATE_SCENARIO: dict = {
    "name": "issue_5303_promotion_probe",
    "map_id": "classic_cross_trap",
    "simulation_config": {"max_episode_steps": 30, "ped_density": 0.0},
    "metadata": {"archetype": "promotion_preflight_probe"},
    "seeds": [7],
    "single_pedestrians": [
        {
            "id": CONTRACT_PEDESTRIAN_ID,
            "start": [0.0, 0.0],
            "goal": None,
            "trajectory": [[0.0, 0.0], [1.0, 1.0]],
            "speed_m_s": 1.0,
        }
    ],
}


def _contract_search_space(*, pedestrian_id: str | None) -> SearchSpaceConfig:
    """Build an in-memory search space for the contract self-test."""
    return SearchSpaceConfig(
        start_x=RangeConfig(1.0, 1.0),
        start_y=RangeConfig(2.0, 2.0),
        goal_x=RangeConfig(5.0, 5.0),
        goal_y=RangeConfig(2.0, 2.0),
        spawn_time_s=RangeConfig(0.0, 2.0),
        pedestrian_speed_mps=RangeConfig(1.0, 1.0),
        pedestrian_delay_s=RangeConfig(0.0, 1.5),
        scenario_seed=RangeConfig(7.0, 7.0),
        min_start_goal_distance_m=0.5,
        pedestrian_id=pedestrian_id,
    )


def _run_contract_self_test(*, output_format: str) -> int:
    """Run the built-in positive/negative contract controls.

    Returns:
        Process exit code (0 when the gate behaves; 1 on any contract violation).
    """
    ready = evaluate_preflight(
        search_space=_contract_search_space(pedestrian_id=CONTRACT_PEDESTRIAN_ID),
        template_scenario=CONTRACT_TEMPLATE_SCENARIO,
    )
    inert = evaluate_preflight(
        search_space=_contract_search_space(pedestrian_id=None),
        template_scenario=CONTRACT_TEMPLATE_SCENARIO,
    )

    checks: list[tuple[str, bool, str]] = [
        (
            "positive probe reaches promotion_timing_ready",
            ready.status == "promotion_timing_ready",
            ready.status,
        ),
        (
            "positive probe materializes a non-null pedestrian identity",
            ready.materialized_pedestrian_id == CONTRACT_PEDESTRIAN_ID,
            str(ready.materialized_pedestrian_id),
        ),
        (
            "positive probe populates a single_pedestrians entry",
            ready.single_pedestrian_populated,
            str(ready.single_pedestrian_populated),
        ),
        (
            "positive probe populates a pedestrian route",
            ready.pedestrian_route_populated,
            str(ready.pedestrian_route_populated),
        ),
        (
            "every frozen timing dimension is runtime-effective",
            all(probe.status == "effective" for probe in ready.dimensions),
            ", ".join(f"{p.name}={p.status}" for p in ready.dimensions),
        ),
        (
            "negative control (no pedestrian.id) is rejected as blocked_no_pedestrian",
            inert.status == "blocked_no_pedestrian",
            inert.status,
        ),
    ]
    all_ok = all(passed for _, passed, _ in checks)

    if output_format == "json":
        payload = {
            "contract": "issue-5303-search-promotion.v1",
            "ok": all_ok,
            "checks": [
                {"check": name, "passed": passed, "observed": observed}
                for name, passed, observed in checks
            ],
            "positive": to_dict(ready),
            "negative": to_dict(inert),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("# Issue #5303 search-promotion contract self-test")
        print("")
        print("- Contract: `issue-5303-search-promotion.v1`")
        print(f"- Gate behaves: {all_ok}")
        print("")
        print("## Checks")
        print("")
        for name, passed, observed in checks:
            mark = "PASS" if passed else "FAIL"
            print(f"- [{mark}] {name} (observed: `{observed}`)")
        print("")
        print("## Positive probe (promotion-ready search space)")
        print("")
        print(render_markdown(ready))
        print("## Negative control (no pedestrian.id; PR #6291 failure mode)")
        print("")
        print(render_markdown(inert))

    return 0 if all_ok else 1


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-space",
        type=Path,
        default=None,
        help="Path to a search-space YAML to probe. With --scenario-template, probes that "
        "pair instead of running the built-in contract self-test.",
    )
    parser.add_argument(
        "--scenario-template",
        type=Path,
        default=None,
        help="Path to a scenario-template YAML used with --search-space.",
    )
    parser.add_argument(
        "--pedestrian-id",
        type=str,
        default=None,
        help=(
            "Optional pedestrian identity assertion for a --search-space probe; it must "
            "match the ID declared in the search space."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format (default: %(default)s).",
    )
    parser.add_argument(
        "--fail-on-blocked",
        action="store_true",
        help="Exit non-zero unless a --search-space probe reaches 'promotion_timing_ready'.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the contract self-test or probe a specific config pair.

    Returns:
        Process exit code (0 on success; 1 when the gate misbehaves or a probed config is
        blocked with ``--fail-on-blocked``; 2 on load/parse failure).
    """
    args = _parse_args(argv)

    if bool(args.search_space) != bool(args.scenario_template):
        print(
            "error: --search-space and --scenario-template must be provided together",
            file=sys.stderr,
        )
        return 2

    if args.search_space is None:
        return _run_contract_self_test(output_format=args.format)

    try:
        preflight: SearchPromotionPreflight = evaluate_preflight_from_files(
            search_space_path=args.search_space,
            scenario_template_path=args.scenario_template,
            pedestrian_id=args.pedestrian_id,
        )
    except SearchPromotionPreflightError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(to_dict(preflight), indent=2, sort_keys=True))
    else:
        print(render_markdown(preflight))

    if args.fail_on_blocked and not preflight.promotion_ready:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
