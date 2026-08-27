"""Effective algorithm-branch coverage for release diagnostics (issue #7937).

A scenario-adaptive planner arm may declare ``scenario_algo_overrides`` that
switch the effective algorithm for specific scenarios (e.g.
``francis2023_leave_group -> orca``).  A diagnostic witness set must cover
every non-default branch before a full campaign consumes compute; otherwise a
validator/provenance mismatch is only discovered at the publication gate.

This module owns the deterministic branch enumeration and the
witness-coverage check so the stress-smoke, runtime-smoke, and
release-acceptance contracts share one effective-algorithm contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Bounded vocabulary of diagnostic witness kinds accepted by the coverage check.
WITNESS_KINDS = frozenset({"scenario_cell", "episode_row", "diagnostic_row"})


def enumerate_effective_branches(
    candidate_payload: Mapping[str, Any],
    *,
    allowed_scenario_ids: set[str] | None = None,
) -> list[dict[str, str]]:
    """Enumerate the effective ``(arm, scenario, algorithm)`` branches.

    A candidate config declares a default algorithm (``algo`` or a planner
    key) plus optional ``scenario_algo_overrides``.  Each override produces one
    non-default branch; the default algorithm is not listed unless a scenario
    is both configured and not overridden.

    Returns:
        Sorted list of branch records with ``arm``, ``scenario``, and
        ``algorithm`` keys.
    """
    arm = str(candidate_payload.get("id") or candidate_payload.get("name") or "unknown")
    default_algo = str(
        candidate_payload.get("algo")
        or candidate_payload.get("algorithm")
        or candidate_payload.get("planner")
        or "default"
    )
    raw_overrides = candidate_payload.get("scenario_algo_overrides")
    branches: list[dict[str, str]] = []
    if not isinstance(raw_overrides, Mapping):
        return branches
    for raw_scenario_id, raw_override in raw_overrides.items():
        scenario_id = str(raw_scenario_id).strip()
        if allowed_scenario_ids is not None and scenario_id not in allowed_scenario_ids:
            continue
        if not isinstance(raw_override, Mapping):
            continue
        algorithm = str(raw_override.get("algo") or default_algo).strip() or default_algo
        branches.append(
            {
                "arm": arm,
                "scenario": scenario_id,
                "algorithm": algorithm,
            }
        )
    branches.sort(key=lambda branch: (branch["arm"], branch["scenario"], branch["algorithm"]))
    return branches


def check_witness_coverage(
    branches: list[dict[str, str]],
    witnesses: list[Mapping[str, Any]],
    *,
    witness_kinds: frozenset[str] = WITNESS_KINDS,
) -> list[str]:
    """Return problems when a non-default branch lacks a diagnostic witness.

    A witness row binds a branch when its ``arm``, ``scenario``, and
    ``algorithm`` fields match exactly, or when it carries a ``branch_key`` of
    the form ``arm|scenario|algorithm``.  Witness rows of an unknown kind are
    ignored (not counted as coverage).

    Returns:
        Problem strings; empty when every branch has at least one witness.
    """
    covered: set[tuple[str, str, str]] = set()
    for witness in witnesses:
        kind = str(witness.get("kind") or "episode_row")
        if kind not in witness_kinds:
            continue
        branch_key = witness.get("branch_key")
        if isinstance(branch_key, str) and branch_key:
            parts = branch_key.split("|")
            if len(parts) == 3:
                covered.add((parts[0], parts[1], parts[2]))
                continue
        arm = str(witness.get("arm") or "")
        scenario = str(witness.get("scenario") or "")
        algorithm = str(witness.get("algorithm") or "")
        if arm and scenario and algorithm:
            covered.add((arm, scenario, algorithm))

    problems: list[str] = []
    for branch in branches:
        key = (branch["arm"], branch["scenario"], branch["algorithm"])
        if key not in covered:
            problems.append(
                f"missing diagnostic witness for effective branch "
                f"{branch['arm']}|{branch['scenario']}|{branch['algorithm']}"
            )
    return problems
