#!/usr/bin/env python3
"""Fail-closed readiness gate for the ScenarioBelief drop-vs-retain runner (#3556).

Plain-language summary: issue #3471 (PR #3553) produced a *diagnostic*-tier finding that
**dropping** uncertain agents raises unsafe commitment while **retaining** them stays near an
oracle baseline. Issue #3556 wants to promote that contrast toward nominal benchmark evidence by
running the same three belief modes through the real benchmark runner. Before any such real run
burns compute, the runner *inputs* must be pinned and the planner's uncertainty contract must
**fail closed** on unsupported paths -- otherwise a real campaign can silently run a planner that
ignores the uncertainty sidecar and the drop-vs-retain contrast becomes meaningless.

This is the cheap CPU-only pre-run gate. It does **not** execute the benchmark matrix, roll
episodes, or interpret outcomes. It only verifies that:

* the predeclared config pins the exact three belief modes (oracle / uncertain_retained /
  uncertain_dropped) needed for the drop-vs-retain contrast plus the near-safe oracle baseline;
* the seed matrix is explicitly pinned (a non-empty list of unique integers), not silently
  defaulted by the runner's fallback;
* every controlled episode parameter is pinned in the config and geometrically valid;
* the uncertainty-consuming planner key (``stream_gap``) is supported and actually consumes the
  uncertainty sidecar for the dropped mode;
* an unsupported planner key **fails closed** (no uncertainty consumed).

It is a thin orchestrator: it **composes** the canonical runner
(``run_scenario_belief_episode_safety_issue_3471``) and the planner-projection contract owner
(``robot_sf.planner.scenario_belief_adapter``); it never redefines the mode set, the parameter
schema, or the fail-closed semantics.

Exit code ``0`` means the runner inputs are pinned and the fail-closed contract holds (safe to
proceed to a real run). Exit ``1`` means at least one readiness check failed (do not run). Exit
``2`` means the check itself could not be evaluated (e.g. missing/unparseable config).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Make the repo root importable when run as a bare script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Canonical contract owners: the planner-projection fail-closed semantics live here.
from robot_sf.planner.scenario_belief_adapter import (  # noqa: E402
    SUPPORTED_UNCERTAINTY_PLANNER_KEYS,
    project_scenario_belief_for_planner,
)

# Canonical runner owner: the mode set, parameter schema, and belief construction live here.
from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (  # noqa: E402
    ISSUE as RUNNER_ISSUE,
)
from scripts.validation.run_scenario_belief_episode_safety_issue_3471 import (  # noqa: E402
    MODES,
    PLANNER_KEY,
    EpisodeParams,
    build_belief_for_mode,
    build_initial_state,
    load_config,
)

SCHEMA_VERSION = "scenario-belief-runner-readiness.v1"
ISSUE = 3556

#: Default predeclared config promoted toward the real runner by #3556.
DEFAULT_CONFIG = (
    _REPO_ROOT / "configs" / "benchmarks" / "scenario_belief_episode_safety_issue_3471.yaml"
)

#: The two modes whose contrast is the entire point of the study; both must be pinned.
REQUIRED_CONTRAST_MODES = ("uncertain_retained", "uncertain_dropped")

#: A planner key guaranteed not to consume the uncertainty sidecar, used to prove fail-closed.
_UNSUPPORTED_PROBE_KEY = "preflight_unsupported_probe"

#: Episode-parameter fields that must be strictly positive for a valid controlled crossing.
_POSITIVE_PARAM_FIELDS = (
    "max_steps",
    "dt",
    "robot_radius",
    "ped_radius",
    "near_miss_margin",
    "ped_cross_speed",
)


@dataclass
class ReadinessCheck:
    """One named readiness assertion and its outcome."""

    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the check."""
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class ReadinessReport:
    """Aggregate readiness verdict for the drop-vs-retain runner inputs."""

    checks: list[ReadinessCheck] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> bool:
        """Record one check and return its pass/fail so callers can short-circuit."""
        self.checks.append(ReadinessCheck(name=name, passed=passed, detail=detail))
        return passed

    @property
    def ready(self) -> bool:
        """True only when every recorded check passed."""
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[ReadinessCheck]:
        """Return the failed checks (empty when ready)."""
        return [c for c in self.checks if not c.passed]

    def as_dict(self) -> dict[str, Any]:
        """Return the structured, JSON-serializable readiness report."""
        return {
            "schema_version": SCHEMA_VERSION,
            "issue": ISSUE,
            "runner_issue": RUNNER_ISSUE,
            "planner_key": PLANNER_KEY,
            "ready": self.ready,
            "checks": [c.as_dict() for c in self.checks],
            "failed_checks": [c.name for c in self.failures],
            "claim_boundary": (
                "Input-pinning and fail-closed readiness gate only; does not run the benchmark "
                "matrix, roll episodes, or interpret drop-vs-retain outcomes."
            ),
        }


def _check_belief_modes(report: ReadinessReport, raw: dict[str, Any]) -> None:
    """Verify the config pins exactly the three modes the drop-vs-retain contrast needs."""
    declared = raw.get("belief_modes")
    if not isinstance(declared, list) or not declared:
        report.add(
            "belief_modes_pinned",
            False,
            "config is missing a non-empty 'belief_modes' list",
        )
        return
    declared_set = set(declared)
    unknown = declared_set - set(MODES)
    missing = set(MODES) - declared_set
    if unknown or missing:
        report.add(
            "belief_modes_pinned",
            False,
            f"belief_modes must equal {sorted(MODES)}; unknown={sorted(unknown)} "
            f"missing={sorted(missing)}",
        )
    else:
        report.add(
            "belief_modes_pinned",
            True,
            f"pinned exactly {sorted(MODES)}",
        )
    # The contrast pair plus the near-safe oracle baseline are individually load-bearing.
    contrast_missing = [m for m in (*REQUIRED_CONTRAST_MODES, "oracle") if m not in declared_set]
    report.add(
        "contrast_modes_present",
        not contrast_missing,
        "drop-vs-retain contrast + oracle baseline all pinned"
        if not contrast_missing
        else f"missing required modes: {contrast_missing}",
    )


def _check_seeds(report: ReadinessReport, raw: dict[str, Any]) -> None:
    """Verify the seed matrix is explicitly pinned, not left to the runner's fallback."""
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        report.add(
            "seeds_pinned",
            False,
            "config must pin a non-empty 'seeds' list (runner fallback is not a pinned matrix)",
        )
        return
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        report.add("seeds_pinned", False, "all seeds must be integers")
        return
    if len(set(seeds)) != len(seeds):
        report.add("seeds_pinned", False, "seed matrix contains duplicates")
        return
    report.add("seeds_pinned", True, f"{len(seeds)} unique integer seeds pinned")


def _check_params(report: ReadinessReport, raw: dict[str, Any], params: EpisodeParams) -> None:
    """Verify every episode parameter is pinned in the config and geometrically valid."""
    declared = raw.get("params")
    if not isinstance(declared, dict):
        report.add("params_pinned", False, "config is missing a 'params' mapping")
        return
    field_names = {f.name for f in fields(EpisodeParams)}
    unpinned = sorted(field_names - set(declared))
    if unpinned:
        report.add(
            "params_pinned",
            False,
            f"these episode params rely on runner defaults instead of being pinned: {unpinned}",
        )
    else:
        report.add("params_pinned", True, f"all {len(field_names)} episode params pinned")

    # Geometric validity: strictly-positive fields, an achievable goal, and a contested corridor.
    problems: list[str] = []
    for name in _POSITIVE_PARAM_FIELDS:
        if getattr(params, name) <= 0:
            problems.append(f"{name} must be > 0 (got {getattr(params, name)})")
    if params.goal_x <= params.start_x:
        problems.append(f"goal_x ({params.goal_x}) must be > start_x ({params.start_x})")
    if not params.start_x < params.corridor_x < params.goal_x:
        problems.append(
            f"corridor_x ({params.corridor_x}) must lie between start_x ({params.start_x}) "
            f"and goal_x ({params.goal_x}) so the crosser actually contests the path"
        )
    report.add(
        "params_valid",
        not problems,
        "controlled-crossing geometry is valid" if not problems else "; ".join(problems),
    )


def _check_planner_contract(report: ReadinessReport, params: EpisodeParams) -> None:
    """Verify the planner key is supported, consumes uncertainty, and fails closed otherwise.

    Builds a single synthetic ``uncertain_dropped`` belief (one step, no episode roll) and
    projects it twice: once under the real planner key (must be ``compatible`` and consume the
    sidecar) and once under an unsupported key (must ``fail_closed`` and consume nothing). This is
    the core "fail closed before real runs" guarantee.
    """
    report.add(
        "planner_supported",
        PLANNER_KEY in SUPPORTED_UNCERTAINTY_PLANNER_KEYS,
        f"runner planner key {PLANNER_KEY!r} in supported set {sorted(SUPPORTED_UNCERTAINTY_PLANNER_KEYS)}"
        if PLANNER_KEY in SUPPORTED_UNCERTAINTY_PLANNER_KEYS
        else f"runner planner key {PLANNER_KEY!r} is not in {sorted(SUPPORTED_UNCERTAINTY_PLANNER_KEYS)}",
    )

    state = build_initial_state(seed=101, params=params)
    belief = build_belief_for_mode(state, "uncertain_dropped", params)

    supported = project_scenario_belief_for_planner(belief, planner_key=PLANNER_KEY)
    consumes = (
        supported.compatibility.get("status") == "compatible"
        and supported.compatibility.get("uncertainty_consumed") is True
    )
    report.add(
        "dropped_mode_consumes_uncertainty",
        consumes,
        "dropped mode consumes the uncertainty sidecar (gate is exercised)"
        if consumes
        else f"dropped mode did not consume uncertainty: {supported.compatibility}",
    )

    unsupported = project_scenario_belief_for_planner(belief, planner_key=_UNSUPPORTED_PROBE_KEY)
    fails_closed = (
        unsupported.compatibility.get("status") == "fail_closed"
        and unsupported.compatibility.get("uncertainty_consumed") is False
    )
    report.add(
        "unsupported_planner_fails_closed",
        fails_closed,
        "unsupported planner key fails closed (no uncertainty consumed)"
        if fails_closed
        else f"unsupported planner key did not fail closed: {unsupported.compatibility}",
    )


def check_runner_readiness(config_path: Path) -> ReadinessReport:
    """Evaluate all drop-vs-retain runner-readiness checks for ``config_path``.

    Raises:
        FileNotFoundError: if the config file does not exist.
        ValueError / yaml.YAMLError: if the config cannot be parsed into the expected shape.
    """
    import yaml

    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path}: expected a YAML mapping at the top level")

    # Resolve the pinned matrix/params through the canonical runner loader so the readiness
    # check evaluates exactly what the real run would consume.
    _seeds, params = load_config(config_path)

    report = ReadinessReport()
    _check_belief_modes(report, raw)
    _check_seeds(report, raw)
    _check_params(report, raw, params)
    _check_planner_contract(report, params)
    return report


def _render_human(report: ReadinessReport, config_path: Path) -> str:
    """Render a human-readable readiness report block."""
    lines = [
        f"readiness gate: ScenarioBelief drop-vs-retain runner (issue #{ISSUE})",
        f"config: {config_path}",
    ]
    for check in report.checks:
        marker = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{marker}] {check.name}: {check.detail}")
    if report.ready:
        lines.append("result: READY — inputs pinned and fail-closed contract holds.")
    else:
        failed = ", ".join(c.name for c in report.failures)
        lines.append(f"result: NOT READY — failed checks: {failed}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="preflight_scenario_belief_runner_readiness_issue_3556.py",
        description=(
            "CPU-only fail-closed readiness gate for the ScenarioBelief drop-vs-retain runner. "
            "Exit 0 = inputs pinned and fail-closed contract holds; 1 = a readiness check failed; "
            "2 = the check could not be evaluated. Does not run the benchmark matrix."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Predeclared scenario/seed config to gate (defaults to the #3471 pinned config).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine-readable JSON report to stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the readiness gate. Returns the process exit code (0 ready, 1 not ready, 2 error)."""
    import yaml

    args = _parse_args(argv)
    # Narrow fail-closed surface: catch the input/parse/contract errors the readiness
    # evaluation can realistically raise (missing or malformed config, bad YAML, type/value
    # mismatches while pinning the matrix or building the probe belief) and report exit 2.
    # Genuinely unexpected errors are intentionally *not* swallowed so they surface as a crash.
    try:
        report = check_runner_readiness(args.config)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        ImportError,
        yaml.YAMLError,
    ) as exc:
        message = f"readiness evaluation failed for {args.config}: {exc}"
        if args.json:
            print(json.dumps({"issue": ISSUE, "ready": False, "error": message}, indent=2))
        else:
            print(f"error: {message}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(_render_human(report, args.config))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
